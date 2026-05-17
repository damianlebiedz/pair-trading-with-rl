"""Script to generate Sub-Period (Quarterly) OOS performance report for Baseline and Agent 2."""

import sys
import pandas as pd
from pathlib import Path
from datetime import datetime
import yaml

from modules.performance.stats import calculate_stats
from modules.core.enums import Interval
from modules.utils.logger import get_logger

logger = get_logger(__name__)

STRATEGIES = {
    "Baseline": "Winners Results/BASELINE OOS 10x",
    "Agent 2": "Winners Results/RL OOS 10x",
}

QUARTERS = {
    "Q1": ("2025-01-01", "2025-03-31"),
    "Q2": ("2025-04-01", "2025-06-30"),
    "Q3": ("2025-07-01", "2025-09-30"),
    "Q4": ("2025-10-01", "2025-12-31"),
}

TITLE = "Sub-Period Out-Of-Sample Performance of the Baseline and Agent 2 (2025)."

SELECTED_METRICS = [
    "cagr",
    "volatility_annual",
    "max_drawdown",
    "sharpe_ratio_annual",
    "sortino_ratio_annual",
    "calmar_ratio",
]

RENAME_MAP = {
    "cagr": "CAGR",
    "volatility_annual": "Annual Volatility",
    "max_drawdown": "Max Drawdown",
    "sharpe_ratio_annual": "Sharpe Ratio (Ann.)",
    "sortino_ratio_annual": "Sortino Ratio (Ann.)",
    "calmar_ratio": "Calmar Ratio",
}

current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
sys.path.append(str(project_root))


def load_strategy_data(
    base_dir: Path, strategy_name: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    strat_dir = base_dir / strategy_name
    if not strat_dir.exists():
        return None, None, None

    returns_files = list(strat_dir.glob("returns_*.parquet"))
    df_returns = pd.read_parquet(returns_files[0]) if returns_files else None

    exec_files = list(strat_dir.glob("exec_logger_*.parquet"))
    df_exec = pd.read_parquet(exec_files[0]) if exec_files else None

    stats_files = list(strat_dir.glob("stats_*.parquet"))
    df_stats = pd.read_parquet(stats_files[0]) if stats_files else None
    if df_stats is not None and "metric" in df_stats.columns:
        df_stats = df_stats.set_index("metric")

    return df_returns, df_exec, df_stats


def get_run_config(base_dir: Path, strategy_name: str) -> dict:
    config_path = base_dir / strategy_name / ".hydra" / "config.yaml"
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}


def slice_and_rebase_data(
    df_ret: pd.DataFrame,
    df_exec: pd.DataFrame,
    start_date: str,
    end_date: str,
    initial_cash: float,
):
    df_ret_q = df_ret.loc[start_date:f"{end_date} 23:59:59"].copy()

    if df_exec is not None:
        df_exec_q = df_exec.loc[start_date:f"{end_date} 23:59:59"].copy()
    else:
        df_exec_q = pd.DataFrame()

    if df_ret_q.empty:
        return None, None

    rets = df_ret_q["equity"].pct_change().fillna(0)
    rets.iloc[0] = 0.0

    df_ret_q["equity"] = initial_cash * (1 + rets).cumprod()
    df_ret_q["total_net_pnl"] = df_ret_q["equity"] - initial_cash

    return df_ret_q, df_exec_q


def format_value(metric_name: str, val: float) -> str:
    if pd.isna(val) or val == "-":
        return "-"
    if metric_name in ["CAGR", "Annual Volatility", "Max Drawdown"]:
        return f"{val:.2%}".replace("%", "\\%")
    else:
        return f"{val:.4f}"


def generate_sub_period_report():
    results_dir = project_root / "results"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_output_dir = results_dir / f"sub_period_report_{timestamp}"
    report_output_dir.mkdir(parents=True, exist_ok=True)

    columns = []
    for q_name in QUARTERS.keys():
        for strat in STRATEGIES.keys():
            columns.append(f"{strat} ({q_name})")

    table_data = {
        m_name: {col: "-" for col in columns} for m_name in RENAME_MAP.values()
    }

    first_strat_folder = next(iter(STRATEGIES.values()))
    config = get_run_config(results_dir, first_strat_folder)
    initial_cash = float(config.get("market", {}).get("initial_cash", 100000))
    risk_free_rate = float(config.get("market", {}).get("risk_free_rate_annual", 0.0))

    logger.info("Processing sub-periods data...")

    for strat_name, folder in STRATEGIES.items():
        logger.info(f"Loading full data for {strat_name}...")
        df_ret_full, df_exec_full, _ = load_strategy_data(results_dir, folder)

        if df_ret_full is None:
            logger.error(f"Could not load data for {strat_name} from {folder}")
            continue

        for q_name, (q_start, q_end) in QUARTERS.items():
            logger.info(f"Calculating stats for {strat_name} - {q_name}...")

            df_ret_q, df_exec_q = slice_and_rebase_data(
                df_ret_full, df_exec_full, q_start, q_end, initial_cash
            )

            col_name = f"{strat_name} ({q_name})"

            if df_ret_q is not None and len(df_ret_q) > 1:
                try:
                    stats = calculate_stats(
                        df_ret_q, df_exec_q, initial_cash, Interval.H1, risk_free_rate
                    )
                    net_stats = stats["net"]

                    for m_key, m_name in RENAME_MAP.items():
                        val = net_stats.get(m_key, None)
                        table_data[m_name][col_name] = format_value(m_name, val)
                except Exception as e:
                    logger.warning(f"Failed to calculate stats for {col_name}: {e}")
            else:
                logger.warning(f"Not enough data for {col_name}")

    logger.info("Generating hierarchical LaTeX table...")

    latex_cols_format = "l" + "*{8}{>{\\centering\\arraybackslash}X}"

    header_l1 = "Metric"
    for q_name in QUARTERS.keys():
        header_l1 += f" & \\multicolumn{{2}}{{c}}{{{q_name} 2025}}"

    header_l2 = " & " + " & ".join(
        ["Baseline" if "Baseline" in c else "Agent 2" for c in columns]
    )

    rows_latex = ""
    for m_key in SELECTED_METRICS:
        m_name = RENAME_MAP[m_key]

        row_vals = []
        for q_name in QUARTERS.keys():
            row_vals.append(table_data[m_name][f"Baseline ({q_name})"])
            row_vals.append(table_data[m_name][f"Agent 2 ({q_name})"])

        formatted_row = " & ".join(row_vals)
        rows_latex += f"        {m_name} & {formatted_row} \\\\\n"

        if m_name in ["Max Drawdown"]:
            rows_latex += "        \\addlinespace[6pt]\n"

    latex_content = f"""\\begin{{table}}[H]
    \\centering
    \\footnotesize
    \\renewcommand{{\\arraystretch}}{{1.2}}
    \\setlength{{\\tabcolsep}}{{2pt}}
    \\caption{{{TITLE}}}
    \\label{{tab:oos-sub-period}}
    \\vspace{{6pt}}
    \\begin{{tabularx}}{{\\linewidth}}{{{latex_cols_format}}}
    \\toprule
        {header_l1} \\\\ 
        {header_l2} \\\\ 
    \\midrule
{rows_latex}    \\bottomrule
    \\end{{tabularx}}

    \\vspace{{10pt}}
    \\justifying \\noindent \\scriptsize \\textit{{Note: Performance metrics isolated for each calendar quarter of 2025. All metrics are calculated by rebasing the initial capital at the start of each period to ensure independent evaluation. Baseline and Agent 2 utilize 0.05\\% fees and 10x leverage.}}
\\end{{table}}
"""

    output_file = report_output_dir / "sub_period_table.tex"
    with open(output_file, "w") as f:
        f.write(latex_content)

    logger.info(f"Sub-period report generated successfully in: {report_output_dir}")
    print(f"\nSaved LaTeX table to: {output_file}")


if __name__ == "__main__":
    generate_sub_period_report()
