"""Script to perform Block Bootstrap significance tests for RL vs Baseline."""

import calendar
import sys
import numpy as np
from pathlib import Path

from helpers.analysis_scripts.generate_multi_report import load_strategy_data

_current_file = Path(__file__).resolve()
_project_root = _current_file.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

project_root = _project_root

STRATEGIES = {
    "Baseline": "Winners Results/BASELINE OOS 10x",
    "Agent 2": "Winners Results/RL OOS 10x",
}
BLOCK_SIZE = 168
N_BOOTSTRAP = 10000
INITIAL_CASH = 100000


def get_annualization_params(df):
    if df.empty:
        return 365 * 24

    start_year = df.index[0].year
    last_ts = df.index[-1]
    end_year = last_ts.year
    if last_ts.month == 1 and last_ts.day == 1:
        end_year -= 1
    if end_year < start_year:
        end_year = start_year

    if start_year == end_year:
        days_in_year = 366 if calendar.isleap(start_year) else 365
    else:
        days_in_year = 365.25

    steps_per_day = 24
    return steps_per_day * days_in_year


def calculate_custom_metrics(returns, periods_per_year, risk_free_rate_annual=0.0):
    actual_periods = len(returns)
    years = actual_periods / periods_per_year

    end_equity = np.prod(1 + returns)
    cagr = (end_equity ** (1 / years)) - 1 if years > 0 else 0

    period_volatility = np.std(returns, ddof=1)
    annual_volatility = period_volatility * np.sqrt(periods_per_year)

    downside_returns = returns[returns < 0]
    if len(downside_returns) > 0:
        period_downside_std = np.std(downside_returns, ddof=1)
        annual_downside_std = period_downside_std * np.sqrt(periods_per_year)
    else:
        annual_downside_std = 0

    sharpe = (
        (cagr - risk_free_rate_annual) / annual_volatility
        if annual_volatility > 0
        else 0
    )
    sortino = (
        (cagr - risk_free_rate_annual) / annual_downside_std
        if annual_downside_std > 0
        else 0
    )

    return sharpe, sortino


def block_bootstrap_test_combined(ret1, ret2, ppy, block_size, n_boot):
    n = len(ret1)
    sh1_orig, s1_orig = calculate_custom_metrics(ret1, ppy)
    sh2_orig, s2_orig = calculate_custom_metrics(ret2, ppy)

    diff_sh_orig = sh2_orig - sh1_orig
    diff_s_orig = s2_orig - s1_orig

    diffs_boot_sh = np.zeros(n_boot)
    diffs_boot_s = np.zeros(n_boot)

    for i in range(n_boot):
        start_indices = np.random.randint(0, n, size=n // block_size + 1)
        boot_idx = np.concatenate(
            [(np.arange(start, start + block_size) % n) for start in start_indices]
        )[:n]

        boot_ret1 = ret1[boot_idx]
        boot_ret2 = ret2[boot_idx]

        sh1_b, s1_b = calculate_custom_metrics(boot_ret1, ppy)
        sh2_b, s2_b = calculate_custom_metrics(boot_ret2, ppy)

        diffs_boot_sh[i] = sh2_b - sh1_b
        diffs_boot_s[i] = s2_b - s1_b

    p_val_sh = np.mean(diffs_boot_sh <= 0)
    p_val_s = np.mean(diffs_boot_s <= 0)

    ci_sh = (np.percentile(diffs_boot_sh, 2.5), np.percentile(diffs_boot_sh, 97.5))
    ci_s = (np.percentile(diffs_boot_s, 2.5), np.percentile(diffs_boot_s, 97.5))

    return (sh1_orig, sh2_orig, diff_sh_orig, p_val_sh, ci_sh), (
        s1_orig,
        s2_orig,
        diff_s_orig,
        p_val_s,
        ci_s,
    )


def run_tests():
    results_dir = project_root / "results"
    df_ret_base, _, _ = load_strategy_data(results_dir, STRATEGIES["Baseline"])
    df_ret_agent, _, _ = load_strategy_data(results_dir, STRATEGIES["Agent 2"])

    if df_ret_base is None or df_ret_agent is None:
        return

    common_idx = df_ret_base.index.intersection(df_ret_agent.index)
    periods_per_year = get_annualization_params(df_ret_base.loc[common_idx])

    pnl_base = df_ret_base.loc[common_idx, "total_net_pnl"].iloc[1:]
    equity_base = pnl_base + INITIAL_CASH
    ret_base = equity_base.pct_change(fill_method=None).dropna().values

    pnl_agent = df_ret_agent.loc[common_idx, "total_net_pnl"].iloc[1:]
    equity_agent = pnl_agent + INITIAL_CASH
    ret_agent = equity_agent.pct_change(fill_method=None).dropna().values

    sharpe_res, sortino_res = block_bootstrap_test_combined(
        ret_base, ret_agent, periods_per_year, BLOCK_SIZE, N_BOOTSTRAP
    )

    for name, res in [("Sharpe", sharpe_res), ("Sortino", sortino_res)]:
        print(f"\n--- {name} Ratio Analysis ---")
        print(f"Baseline: {res[0]:.4f}")
        print(f"Agent 2:  {res[1]:.4f}")
        print(f"Diff:     {res[2]:.4f}")
        print(f"p-value:  {res[3]:.4f}")

    generate_latex_table(sortino_res, sharpe_res)


def generate_latex_table(s_res, sh_res):
    latex = (
        r"""\begin{table}[H]
    \centering\footnotesize\renewcommand{\arraystretch}{1.2}
    \caption{Out-Of-Sample Statistical Significance Testing of Performance Differences (Block Bootstrap, 10 000 iterations, block = 168h, fixed seed = 42).}
    \label{tab:stat-sig-tests}
    \begin{tabular}{lccccc}
    \toprule
    Metric & Baseline & Agent 2 & Difference ($\Delta$) & 95\% CI of Diff. & p-value \\
    \midrule
    Sharpe Ratio & """
        + f"{sh_res[0]:.4f} & {sh_res[1]:.4f} & {sh_res[2]:.4f} & [{sh_res[4][0]:.4f}, {sh_res[4][1]:.4f}] & \\textbf{{{sh_res[3]:.4f}}}"
        + r""" \\
    Sortino Ratio & """
        + f"{s_res[0]:.4f} & {s_res[1]:.4f} & {s_res[2]:.4f} & [{s_res[4][0]:.4f}, {s_res[4][1]:.4f}] & \\textbf{{{s_res[3]:.4f}}}"
        + r""" \\
    \bottomrule
    \end{tabular}
    \vspace{6pt}
    \justifying \noindent \scriptsize Note: Baseline: 0.05\% fees, 10x leverage; Agent 2: 0.05\% fees, 10x leverage. Statistical significance computed via stationary circular block bootstrap using the empirical percentile method. CI denotes Confidence Interval. The null hypothesis states that Agent 2 does not outperform the Baseline ($H_0: \Delta \le 0$).
\end{table}
"""
    )
    with open(project_root / "results" / "stat_sig_table.tex", "w") as f:
        f.write(latex)


if __name__ == "__main__":
    np.random.seed(42)
    run_tests()
