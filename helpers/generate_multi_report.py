import json
import sys
import yaml
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
from datetime import datetime

from modules.data_services.data_utils import load_btc_benchmark, load_ewp_benchmark
from modules.performance.stats import calculate_stats
from modules.core.enums import Interval
from modules.utils.logger import get_logger

logger = get_logger(__name__)

STRATEGIES = {
    "WINNER_1_baseline": "Fixed | No Hedge",
    "WINNER_5_hybrid_fixed": "Fixed | Rolling Beta-Hedge",
}

LEVERAGE = 10.0

FONT_SANS = "Arial, sans-serif"
FONT_SERIF = "Times New Roman, serif"

current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
sys.path.append(str(project_root))

SELECTED_METRICS = [
    "cagr",
    "volatility_annual",
    "max_drawdown",
    "win_count",
    "lose_count",
    "win_rate",
    "avg_win",
    "avg_lose",
    "avg_trade_return",
    "avg_trade_duration",
    "sharpe_ratio_annual",
    "sortino_ratio_annual",
    "calmar_ratio_annual",
    "tda_sortino",
]

RENAME_MAP = {
    "cagr": "CAGR",
    "volatility_annual": "Annual Volatility",
    "max_drawdown": "Max Drawdown",
    "win_count": "Win Count",
    "lose_count": "Loss Count",
    "win_rate": "Win Rate",
    "avg_win": "Avg Win",
    "avg_lose": "Avg Loss",
    "avg_trade_return": "Avg Trade Return",
    "avg_trade_duration": "Avg Trade Duration",
    "sharpe_ratio_annual": "Sharpe Ratio",
    "sortino_ratio_annual": "Sortino Ratio",
    "calmar_ratio_annual": "Calmar Ratio",
    "tda_sortino": "TDA-Sortino",
}


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

    stats_files = list(strat_dir.glob("stats_multi_pair_*.parquet"))
    df_stats = pd.read_parquet(stats_files[0]).set_index("metric") if stats_files else None

    return df_returns, df_exec, df_stats


def get_run_config(base_dir: Path, strategy_name: str) -> dict:
    config_path = base_dir / strategy_name / ".hydra" / "config.yaml"
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}


def build_stitched_ewp(base_dir: Path, strategy_name: str, interval: Interval, assets_dict: dict) -> pd.DataFrame:
    strat_dir = base_dir / strategy_name

    iter_dirs = sorted(
        [d for d in strat_dir.iterdir() if d.is_dir() and d.name.isdigit()],
        key=lambda x: int(x.name),
    )

    ewp_dfs = []

    for d in iter_dirs:
        returns_files = list(d.glob("returns_*.parquet"))
        if not returns_files:
            continue

        df = pd.read_parquet(returns_files[0])
        if df.empty:
            continue

        start_date = df.index[0].strftime("%Y-%m-%d")
        end_date = df.index[-1].strftime("%Y-%m-%d")
        iter_num = int(d.name)

        iter_tickers = assets_dict.get(iter_num) or assets_dict.get(str(iter_num))

        if not iter_tickers:
            logger.warning(f"Warning: Lack of tickers for {iter_num} in 'list_of_assets.json'")
            continue

        ewp_period = load_ewp_benchmark(iter_tickers, start_date, end_date, interval)

        col_name = "portfolio_pct" if "portfolio_pct" in ewp_period.columns else "ewp_pct"
        ewp_dfs.append(ewp_period[[col_name]])

    if not ewp_dfs:
        return None

    final_ewp = pd.concat(ewp_dfs).sort_index()
    final_ewp = final_ewp[~final_ewp.index.duplicated(keep="first")]

    col_name = "portfolio_pct" if "portfolio_pct" in final_ewp.columns else "ewp_pct"
    final_ewp["ewp_return"] = (1 + final_ewp[col_name]).cumprod() - 1

    return final_ewp


def generate_comparison_report(strategies_map: dict):
    results_dir = project_root / "results"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_output_dir = results_dir / f"final_paper_comparison_{timestamp}"
    pdf_dir = report_output_dir / "pdfs"
    report_output_dir.mkdir(parents=True, exist_ok=True)
    pdf_dir.mkdir(parents=True, exist_ok=True)

    assets_file = project_root / "config" / "list_of_assets.json"
    list_of_assets = {}
    if assets_file.exists():
        with open(assets_file, "r", encoding="utf-8") as f:
            list_of_assets = json.load(f)
    else:
        logger.error(f"'list_of_assets.json' not found: {assets_file}")

    fig = go.Figure()
    stats_dict = {}

    global_starts = []
    global_ends = []

    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b"]
    color_idx = 0

    first_strategy_folder = None
    common_initial_cash = 100000.0
    common_risk_free = 0.0

    for folder, label in strategies_map.items():
        logger.info(f"Loading strategy: {label} from {folder}")

        if first_strategy_folder is None:
            first_strategy_folder = folder

        run_config = get_run_config(results_dir, folder)
        market_cfg = run_config.get("market", {})

        initial_cash = float(market_cfg.get("initial_cash"))
        risk_free_rate = float(market_cfg.get("risk_free_rate_annual"))

        common_initial_cash = initial_cash
        common_risk_free = risk_free_rate

        df_ret, df_exec, df_stats = load_strategy_data(results_dir, folder)

        if df_ret is None:
            logger.warning(f"Data not found for {label} ({folder}). Skipping...")
            continue

        global_starts.append(df_ret.index[0])
        global_ends.append(df_ret.index[-1])

        pnl_series = (df_ret["total_pnl"] - df_ret["total_fees"]) * LEVERAGE
        ret_series = pnl_series / initial_cash

        fig.add_trace(go.Scatter(
            x=ret_series.index, y=ret_series, mode="lines",
            name=f"{label}", line=dict(color=colors[color_idx % len(colors)], width=2.0),
        ))

        if df_stats is not None:
            stats_dict[label] = df_stats["net"].reindex(SELECTED_METRICS)

        color_idx += 1

    if not global_starts:
        logger.error("No valid strategies loaded. Exiting.")
        return

    full_start = min(global_starts).strftime("%Y-%m-%d")
    full_end = max(global_ends).strftime("%Y-%m-%d")

    empty_ex = pd.DataFrame(columns=["position", "pnl", "fees", "entry_equity"])

    try:
        btc_data = load_btc_benchmark(full_start, full_end, Interval.H1)
        col_btc = "close" if "close" in btc_data.columns else btc_data.columns[0]

        btc_ret = (btc_data[col_btc] / btc_data[col_btc].iloc[0]) - 1

        fig.add_trace(go.Scatter(
            x=btc_ret.index, y=btc_ret, mode="lines", name="BTC B&H",
            line=dict(color="gray", width=1.5, dash="dash"),
        ))

        df_btc = pd.DataFrame(index=btc_ret.index)
        df_btc["total_pnl"] = btc_ret * common_initial_cash
        df_btc["total_net_pnl"] = df_btc["total_pnl"]

        stats_dict["BTC B&H"] = calculate_stats(
            df_btc, empty_ex, common_initial_cash, Interval.H1, common_risk_free
        )["net"].reindex(SELECTED_METRICS)
    except Exception as e:
        logger.error(f"Error during BTC data loading: {e}")

    try:
        ewp_data = build_stitched_ewp(results_dir, first_strategy_folder, Interval.H1, list_of_assets)

        if ewp_data is not None:
            ewp_ret = ewp_data["ewp_return"]

            fig.add_trace(go.Scatter(
                x=ewp_ret.index, y=ewp_ret, mode="lines", name="EWP Portfolio",
                line=dict(color="black", width=1.5)
            ))

            df_ewp = pd.DataFrame(index=ewp_ret.index)
            df_ewp["total_pnl"] = ewp_ret * common_initial_cash
            df_ewp["total_net_pnl"] = df_ewp["total_pnl"]

            stats_dict["EWP Portfolio"] = calculate_stats(
                df_ewp, empty_ex, common_initial_cash, Interval.H1, common_risk_free
            )["net"].reindex(SELECTED_METRICS)
        else:
            logger.error("EWP data not generated.")
    except Exception as e:
        logger.error(f"Error during EWP data loading: {e}")

    fig.update_layout(
        template="plotly_white", font=dict(family=FONT_SANS, color="black"),
        margin=dict(t=50, b=50, l=70, r=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="center", x=0.5, font=dict(size=12)),
        width=900, height=450,
    )

    fig.update_yaxes(title="Cumulative Return", tickformat=".0%", showline=True, linewidth=1, linecolor="black",
                     gridcolor="#e5e5e5", mirror=True, zeroline=True, zerolinecolor="black")
    fig.update_xaxes(showline=True, linewidth=1, linecolor="black", gridcolor="#e5e5e5", mirror=True)

    try:
        pdf_path = pdf_dir / "multi_strategy_comparison.pdf"
        fig.write_image(str(pdf_path), format="pdf", engine="kaleido")
        logger.info(f"PDF saved: {pdf_path}")
    except Exception as e:
        logger.error(f"Error during PDF export: {e}")

    df_stats = pd.DataFrame(stats_dict).rename(index=RENAME_MAP)

    for col in df_stats.columns:
        for idx in df_stats.index:
            val = df_stats.loc[idx, col]
            if pd.notna(val):
                if idx in ["CAGR", "Annual Volatility", "Max Drawdown", "Win Rate", "Avg Trade Return", "Avg Win",
                           "Avg Loss"]:
                    df_stats.loc[idx, col] = f"{val:.2%}"
                elif idx in ["Win Count", "Loss Count"]:
                    df_stats.loc[idx, col] = f"{int(val)}"
                else:
                    df_stats.loc[idx, col] = f"{val:.4f}"
            else:
                df_stats.loc[idx, col] = "-"

    with open(report_output_dir / "comparison_table.tex", "w") as f:
        f.write(df_stats.to_latex(column_format="l" + "r" * len(df_stats.columns), escape=False))

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body {{ font-family: "{FONT_SERIF}"; padding: 30px; max-width: 1000px; margin: auto; background-color: #fcfcfc; }}
            .report-container {{ background-color: white; padding: 30px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }}
            h2, h3 {{ text-align: center; color: #333; }}
            .elsevier-table {{ width: 100%; border-collapse: collapse; margin-top: 20px; font-size: 11pt; }}
            .elsevier-table thead tr {{ border-top: 2px solid black; border-bottom: 1px solid black; }}
            .elsevier-table th, .elsevier-table td {{ padding: 8px; text-align: right; }}
            .elsevier-table td:first-child, .elsevier-table th:first-child {{ text-align: left; font-weight: bold; width: 220px; }}
            .elsevier-table tbody tr:last-child td {{ border-bottom: 2px solid black; }}
            .elsevier-table tbody tr:hover {{ background-color: #f5f5f5; }}
        </style>
    </head>
    <body>
        <div class="report-container">
            <h2>Strategy Performance Comparison</h2>
            <div>{fig.to_html(full_html=False, include_plotlyjs='cdn')}</div>
            <br><br>
            <h3>Table 1: Performance Metrics</h3>
            {df_stats.to_html(classes="elsevier-table", border=0, justify="center")}
        </div>
    </body>
    </html>
    """

    html_path = report_output_dir / "interactive_comparison.html"
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    logger.info(f"Comparison Pipeline completed. Results saved to: {report_output_dir}")


if __name__ == "__main__":
    generate_comparison_report(STRATEGIES)
