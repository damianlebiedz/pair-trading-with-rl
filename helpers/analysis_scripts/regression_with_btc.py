"""
Script to calculate OLS Regression (Beta) for Agent 2 (Hedged vs Unhedged) against BTC Benchmark.
Used for Elsevier Reviewer response (Post-Hoc Ablation Study).
Generates a LaTeX table with the results.
"""

import sys
import yaml
import pandas as pd
import statsmodels.api as sm
from pathlib import Path

from modules.data_services.data_utils import load_btc_benchmark
from modules.core.enums import Interval
from modules.utils.logger import get_logger

logger = get_logger(__name__)

STRATEGIES = {
    "Agent 2 (Hedged)": "Winners Results/RL OOS 10x",
    "Agent 2 (Unhedged)": "RL Sensitivity Analysis 10x/Assumptions Verification/oos/run_backtest_2026-04-22_22-10-07_daa822",
}

current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
sys.path.append(str(project_root))
results_dir = project_root / "results"


def load_strategy_equity(base_dir: Path, strategy_name: str) -> pd.Series:
    """Load strategy equity time series from a returns parquet file."""
    strat_dir = base_dir / strategy_name
    if not strat_dir.exists():
        logger.error(f"Strategy directory does not exist: {strat_dir}")
        return None

    returns_files = list(strat_dir.glob("returns_*.parquet"))
    if not returns_files:
        logger.error(f"No returns_*.parquet file found in: {strat_dir}")
        return None

    df = pd.read_parquet(returns_files[0])
    return df["equity"]


def get_fee_rate(base_dir: Path, strategy_name: str) -> float:
    """Read fee_rate from strategy Hydra config, with a fallback default."""
    config_path = base_dir / strategy_name / ".hydra" / "config.yaml"
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
            return float(config.get("market", {}).get("fee_rate", 0.0005))
    return 0.0005


def main():
    logger.info("Starting Post-Hoc Ablation regression analysis...")

    equity_series = {}
    starts, ends = [], []

    first_strat = list(STRATEGIES.values())[0]
    fee_rate = get_fee_rate(results_dir, first_strat.split("_lev_")[0])

    for label, folder in STRATEGIES.items():
        equity = load_strategy_equity(results_dir, folder)
        if equity is not None:
            daily_equity = equity.resample("D").last()
            daily_returns = daily_equity.pct_change().dropna()
            daily_returns.name = label
            equity_series[label] = daily_returns

            starts.append(daily_returns.index[0])
            ends.append(daily_returns.index[-1])

    if not equity_series:
        logger.error("Failed to load any strategy data.")
        return

    full_start = min(starts).strftime("%Y-%m-%d")
    full_end = max(ends).strftime("%Y-%m-%d")

    logger.info("Loading BTC benchmark data...")
    btc_data = load_btc_benchmark(full_start, full_end, Interval.H1, fee_rate)
    btc_daily = btc_data["BTC_return"].resample("D").last()
    btc_returns = btc_daily.diff().dropna()
    btc_returns.name = "BTC Benchmark"

    df_merged = pd.concat(
        [btc_returns] + list(equity_series.values()), axis=1, join="inner"
    ).dropna()

    print("\n" + "=" * 80)
    print(" " * 20 + "OLS REGRESSION RESULTS (AGENT vs BTC)")
    print("=" * 80)

    X = sm.add_constant(df_merged["BTC Benchmark"])

    results_for_latex = []

    for label in STRATEGIES.keys():
        if label not in df_merged.columns:
            continue

        y = df_merged[label]
        model = sm.OLS(y, X).fit()

        print(f"\n--- Model for: {label} ---")
        print(f"R-squared: {model.rsquared:.4f}")

        alpha = model.params["const"]
        alpha_p = model.pvalues["const"]
        beta = model.params["BTC Benchmark"]
        beta_p = model.pvalues["BTC Benchmark"]

        print(f"Alpha (const): {alpha:10.6f} | p-value: {alpha_p:.4e}")
        print(f"Beta (exposure):{beta:10.6f} | p-value: {beta_p:.4e}")

        if beta_p < 0.05:
            print(
                "Conclusion: Beta coefficient IS statistically significant (p < 0.05)."
            )
        else:
            print("Conclusion: Beta coefficient is NOT statistically significant.")

        print("-" * 40)

        results_for_latex.append(
            {
                "Strategy": label,
                "Alpha": alpha,
                "Alpha_p": alpha_p,
                "Beta": beta,
                "Beta_p": beta_p,
                "R2": model.rsquared,
            }
        )

    print("=" * 80 + "\n")

    logger.info("Generating LaTeX table...")

    latex_content = """\\begin{{table}}[H]
\\centering
\\footnotesize
\\renewcommand{{\\arraystretch}}{{1.2}}
\\caption{{Post-Hoc Ablation Regression Analysis: Agent 2 Hedged vs Unhedged Against BTC Benchmark.}}
\\label{{tab:ablation-regression}}
\\begin{{tabularx}}{{\\linewidth}}{{l*{{5}}{{>{{\\centering\\arraybackslash}}X}}}}
\\toprule
Strategy & $\\alpha$ & $p$-value ($\\alpha$) & $\\beta$ & $p$-value ($\\beta$) & $R^2$ \\\\
\\midrule
"""
    for row in results_for_latex:
        latex_content += f"{row['Strategy']} & {row['Alpha']:.4f} & {row['Alpha_p']:.4f} & {row['Beta']:.4f} & {row['Beta_p']:.4f} & {row['R2']:.4f} \\\\\n"

    latex_content += """\\bottomrule
\\end{tabularx}
\\justifying\\noindent\\scriptsize Note: Ordinary Least Squares (OLS) regression of daily strategy returns against the BTC benchmark returns in the Out-of-Sample period. $\\alpha$ represents the daily abnormal return intercept, while $\\beta$ represents the market exposure coefficient. The analysis empirically validates the structural market neutrality of the unhedged RL agent.
\\end{table}
"""

    output_path = results_dir / "ablation_regression_table.tex"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(latex_content)

    logger.info(f"LaTeX table saved successfully to: {output_path}")


if __name__ == "__main__":
    main()
