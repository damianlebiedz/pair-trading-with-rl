import sys
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
from datetime import datetime

from modules.utils.plots import _get_custom_tickvals
from modules.data_services.data_utils import load_btc_benchmark, load_ewp_benchmark
from modules.performance.stats import calculate_stats
from modules.core.enums import Interval

# ==========================================
STRATEGY = {
    "run_backtest_2026-03-01_19-25-51_c78f35": "Rolling Beta-Hedge",
}

LEVERAGE = 5.0
FEE_MULTIPLIER = 0.5

INITIAL_CASH = 100000.0
RISK_FREE_RATE_ANNUAL = 0.0
# ==========================================

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
    "avg_win_return",
    "avg_lose_return",
    "avg_trade_return",
    "sharpe_ratio_annual",
    "sortino_ratio_annual",
    "calmar_ratio_annual",
]

FORMAT_MAP = {
    "cagr": "{:.2%}",
    "volatility_annual": "{:.2%}",
    "max_drawdown": "{:.2%}",
    "win_rate": "{:.2%}",
    "avg_win_return": "{:.2%}",
    "avg_lose_return": "{:.2%}",
    "avg_trade_return": "{:.2%}",
    "sharpe_ratio_annual": "{:.4f}",
    "sortino_ratio_annual": "{:.4f}",
    "calmar_ratio_annual": "{:.4f}",
    "win_count": "{:.0f}",
    "lose_count": "{:.0f}",
}

RENAME_MAP = {
    "cagr": "CAGR",
    "volatility_annual": "Annual Volatility",
    "max_drawdown": "Max Drawdown",
    "win_count": "Win Count",
    "lose_count": "Lose Count",
    "win_rate": "Win Rate",
    "avg_win_return": "Avg Win",
    "avg_lose_return": "Avg Loss",
    "avg_trade_return": "Avg Trade Return",
    "sharpe_ratio_annual": "Sharpe Ratio",
    "sortino_ratio_annual": "Sortino Ratio",
    "calmar_ratio_annual": "Calmar Ratio",
}


def load_strategy_data(
    base_dir: Path, strategy_name: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    strat_dir = base_dir / strategy_name

    if not strat_dir.exists():
        print(f"Directory not found: {strat_dir}")
        return None, None

    returns_files = list(strat_dir.glob("returns_*.parquet"))
    df_returns = pd.read_parquet(returns_files[0]) if returns_files else None

    exec_files = list(strat_dir.glob("exec_logger_*.parquet"))
    df_exec = pd.read_parquet(exec_files[0]) if exec_files else None

    return df_returns, df_exec


def generate_comparison_report(strategies_input: dict | list) -> None:
    results_dir = project_root / "results"
    report_output_dir = results_dir / "report"
    report_output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    chart_filename = f"one_comparison_chart_{timestamp}.html"
    report_filename = f"one_report_{timestamp}.html"

    strategies_plot_data = {}
    stats_df_dict = {}

    all_dates = []

    if isinstance(strategies_input, list):
        strategies_map = {s: s for s in strategies_input}
    else:
        strategies_map = strategies_input

    leverages_to_test = [1.0]
    if LEVERAGE != 1.0:
        leverages_to_test.append(LEVERAGE)

    for folder, label in strategies_map.items():
        print(f"Loading and processing: {folder}...")
        df_ret, df_exec = load_strategy_data(results_dir, folder)

        if df_ret is not None and not df_ret.empty:
            all_dates.extend(df_ret.index.tolist())
            strategies_plot_data[label] = {}

            for current_lev in leverages_to_test:
                lev_str = f"{current_lev}x"
                label_gross = f"Gross (0%) ({lev_str})"
                label_net_custom = f"Net ({0.01 * FEE_MULTIPLIER:g}%) ({lev_str})"
                label_net_base = f"Net (0.01%) ({lev_str})"

                df_base = df_ret.copy()
                df_base["total_pnl"] = df_base["total_pnl"] * current_lev
                df_base["total_net_pnl"] = df_base["total_pnl"]

                exec_gross = df_exec.copy() if df_exec is not None else pd.DataFrame()
                if not exec_gross.empty:
                    exec_gross["pnl"] = exec_gross["pnl"] * current_lev
                    exec_gross["fees"] = 0.0

                stats_gross = calculate_stats(
                    df=df_base,
                    exec_log_df=exec_gross,
                    initial_cash=INITIAL_CASH,
                    interval=Interval.H1,
                    risk_free_rate_annual=RISK_FREE_RATE_ANNUAL,
                )

                df_net1 = df_base.copy()
                df_net1["total_net_pnl"] = df_base["total_pnl"] - (
                    df_ret["total_fees"] * FEE_MULTIPLIER * current_lev
                )

                exec_net1 = df_exec.copy() if df_exec is not None else pd.DataFrame()
                if not exec_net1.empty:
                    exec_net1["pnl"] = exec_net1["pnl"] * current_lev
                    exec_net1["fees"] = df_exec["fees"] * FEE_MULTIPLIER * current_lev

                stats_net1 = calculate_stats(
                    df=df_net1,
                    exec_log_df=exec_net1,
                    initial_cash=INITIAL_CASH,
                    interval=Interval.H1,
                    risk_free_rate_annual=RISK_FREE_RATE_ANNUAL,
                )

                df_net2 = df_base.copy()
                df_net2["total_net_pnl"] = df_base["total_pnl"] - (
                    df_ret["total_fees"] * 1.0 * current_lev
                )

                exec_net2 = df_exec.copy() if df_exec is not None else pd.DataFrame()
                if not exec_net2.empty:
                    exec_net2["pnl"] = exec_net2["pnl"] * current_lev
                    exec_net2["fees"] = df_exec["fees"] * 1.0 * current_lev

                stats_net2 = calculate_stats(
                    df=df_net2,
                    exec_log_df=exec_net2,
                    initial_cash=INITIAL_CASH,
                    interval=Interval.H1,
                    risk_free_rate_annual=RISK_FREE_RATE_ANNUAL,
                )

                stats_df_dict[(label, label_gross)] = stats_gross["net"].reindex(
                    SELECTED_METRICS
                )
                stats_df_dict[(label, label_net_custom)] = stats_net1["net"].reindex(
                    SELECTED_METRICS
                )
                stats_df_dict[(label, label_net_base)] = stats_net2["net"].reindex(
                    SELECTED_METRICS
                )

                strategies_plot_data[label][label_gross] = (
                    df_base["total_net_pnl"] / INITIAL_CASH
                )
                strategies_plot_data[label][label_net_custom] = (
                    df_net1["total_net_pnl"] / INITIAL_CASH
                )
                strategies_plot_data[label][label_net_base] = (
                    df_net2["total_net_pnl"] / INITIAL_CASH
                )

    if strategies_plot_data:
        global_start = min(all_dates).strftime("%Y-%m-%d")
        global_end = max(all_dates).strftime("%Y-%m-%d")

        btc_data = load_btc_benchmark(
            test_start=global_start, test_end=global_end, interval=Interval.H1
        )
        ewp_data = load_ewp_benchmark(
            tickers=[
                "BTCUSDT",
                "ETHUSDT",
                "BNBUSDT",
                "SOLUSDT",
                "XRPUSDT",
                "ADAUSDT",
                "AVAXUSDT",
                "DOGEUSDT",
                "TRXUSDT",
                "DOTUSDT",
                "LINKUSDT",
                "SHIBUSDT",
                "LTCUSDT",
                "BCHUSDT",
                "UNIUSDT",
                "XLMUSDT",
                "ATOMUSDT",
                "ICPUSDT",
                "FILUSDT",
                "LDOUSDT",
                "APTUSDT",
                "QNTUSDT",
                "ARBUSDT",
                "VETUSDT",
                "MKRUSDT",
                "OPUSDT",
                "NEARUSDT",
                "GRTUSDT",
                "AAVEUSDT",
                "ALGOUSDT",
            ],
            test_start=global_start,
            test_end=global_end,
            interval=Interval.H1,
        )

        fig = go.Figure()
        colors = [
            "#1f77b4",
            "#2ca02c",
            "#d62728",
            "#9467bd",
            "#ff7f0e",
            "#e377c2",
            "#8c564b",
        ]

        first_label = list(strategies_plot_data.keys())[0]
        first_lev_key = list(strategies_plot_data[first_label].keys())[0]
        first_series = strategies_plot_data[first_label][first_lev_key]
        custom_ticks = (
            _get_custom_tickvals(first_series.index)
            if hasattr(first_series, "index")
            else []
        )

        strategy_colors = {
            name: colors[i % len(colors)]
            for i, name in enumerate(strategies_plot_data.keys())
        }

        for current_lev in leverages_to_test:
            lev_str = f"{current_lev}x"
            group_name = f"Leverage {lev_str}"
            is_first_trace_for_group = True

            for name, lev_curves in strategies_plot_data.items():
                color = strategy_colors[name]

                label_gross = f"Gross (0%) ({lev_str})"
                label_net_custom = f"Net ({0.01 * FEE_MULTIPLIER:g}%) ({lev_str})"
                label_net_base = f"Net (0.01%) ({lev_str})"

                w_main = 2.0 if current_lev > 1.0 else 1.0
                w_sub = 1.5 if current_lev > 1.0 else 0.8

                default_visibility = True if current_lev > 1.0 else "legendonly"

                fig.add_trace(
                    go.Scatter(
                        x=lev_curves[label_gross].index,
                        y=lev_curves[label_gross],
                        mode="lines",
                        name=f"{name} - Gross (0%)",
                        legendgroup=group_name,
                        legendgrouptitle=(
                            dict(text=f"<b>{group_name}</b>")
                            if is_first_trace_for_group
                            else None
                        ),
                        line=dict(color=color, width=w_sub, dash="dot"),
                        opacity=1.0,
                        hovertemplate=f"<b>{name} {label_gross}</b>: %{{y:.2%}}<extra></extra>",
                        visible=default_visibility,
                    )
                )
                is_first_trace_for_group = False

                fig.add_trace(
                    go.Scatter(
                        x=lev_curves[label_net_custom].index,
                        y=lev_curves[label_net_custom],
                        mode="lines",
                        name=f"{name} - Net ({0.01 * FEE_MULTIPLIER:g}%)",
                        legendgroup=group_name,
                        line=dict(color=color, width=w_sub, dash="dash"),
                        opacity=1.0,
                        hovertemplate=f"<b>{name} {label_net_custom}</b>: %{{y:.2%}}<extra></extra>",
                        visible=default_visibility,
                    )
                )

                fig.add_trace(
                    go.Scatter(
                        x=lev_curves[label_net_base].index,
                        y=lev_curves[label_net_base],
                        mode="lines",
                        name=f"{name} - Net (0.01%)",
                        legendgroup=group_name,
                        line=dict(color=color, width=w_main),
                        opacity=1.0,
                        hovertemplate=f"<b>{name} {label_net_base}</b>: %{{y:.2%}}<extra></extra>",
                        visible=default_visibility,
                    )
                )

        empty_exec_df = pd.DataFrame(
            columns=["position", "pnl", "fees", "entry_equity"]
        )
        is_first_benchmark = True

        if btc_data is not None and not btc_data.empty:
            if btc_data.index.tz is not None and first_series.index.tz is None:
                btc_data.index = btc_data.index.tz_localize(None)
            elif btc_data.index.tz is None and first_series.index.tz is not None:
                btc_data.index = btc_data.index.tz_localize(first_series.index.tz)

            btc_sub = btc_data.loc[global_start:global_end].copy()

            if not btc_sub.empty:
                col = "close" if "close" in btc_sub.columns else btc_sub.columns[0]
                start_px = btc_sub[col].iloc[0]
                if start_px != 0:
                    btc_ret = (btc_sub[col] / start_px) - 1

                    fig.add_trace(
                        go.Scatter(
                            x=btc_sub.index,
                            y=btc_ret,
                            mode="lines",
                            name="BTC Benchmark",
                            legendgroup="Benchmarks",
                            legendgrouptitle=(
                                dict(text="<b>Benchmarks</b>")
                                if is_first_benchmark
                                else None
                            ),
                            line=dict(color="grey", width=1.5, dash="dot"),
                            opacity=0.6,
                            hovertemplate="<b>BTC Benchmark</b>: %{{y:.2%}}<extra></extra>",
                            visible="legendonly",
                        )
                    )
                    is_first_benchmark = False

                    df_btc = pd.DataFrame(index=btc_sub.index)
                    df_btc["total_pnl"] = btc_ret * INITIAL_CASH
                    df_btc["total_net_pnl"] = df_btc["total_pnl"]

                    btc_stats = calculate_stats(
                        df_btc,
                        empty_exec_df,
                        INITIAL_CASH,
                        Interval.H1,
                        RISK_FREE_RATE_ANNUAL,
                    )

                    btc_stats.loc[["win_count", "lose_count"], "net"] = None
                    btc_stats.loc[["win_count", "lose_count"], "gross"] = None

                    stats_df_dict[("BTC Benchmark", "Buy & Hold")] = btc_stats[
                        "net"
                    ].reindex(SELECTED_METRICS)

        if ewp_data is not None and not ewp_data.empty:
            if ewp_data.index.tz is not None and first_series.index.tz is None:
                ewp_data.index = ewp_data.index.tz_localize(None)

            ewp_ret = ewp_data["ewp_return"]

            fig.add_trace(
                go.Scatter(
                    x=ewp_data.index,
                    y=ewp_ret,
                    mode="lines",
                    name="EWP Benchmark",
                    legendgroup="Benchmarks",
                    legendgrouptitle=(
                        dict(text="<b>Benchmarks</b>") if is_first_benchmark else None
                    ),
                    line=dict(color="black", width=1.5, dash="dot"),
                    opacity=0.6,
                    hovertemplate="<b>EWP Benchmark</b>: %{y:.2%}<extra></extra>",
                    visible="legendonly",
                )
            )

            df_ewp = pd.DataFrame(index=ewp_data.index)
            df_ewp["total_pnl"] = ewp_ret * INITIAL_CASH
            df_ewp["total_net_pnl"] = df_ewp["total_pnl"]

            ewp_stats = calculate_stats(
                df_ewp, empty_exec_df, INITIAL_CASH, Interval.H1, RISK_FREE_RATE_ANNUAL
            )

            ewp_stats.loc[["win_count", "lose_count"], "net"] = None
            ewp_stats.loc[["win_count", "lose_count"], "gross"] = None

            stats_df_dict[("EWP Benchmark", "Buy & Hold")] = ewp_stats["net"].reindex(
                SELECTED_METRICS
            )

        fig.update_layout(
            title=dict(
                text="Comparison of Strategies",
                x=0.5,
                y=0.98,
                yanchor="top",
                font=dict(color="black", size=20),
            ),
            template="plotly_white",
            hovermode="x unified",
            legend=dict(
                orientation="h",
                y=1.00,
                x=0.5,
                xanchor="center",
                yanchor="bottom",
                groupclick="toggleitem",
            ),
            margin=dict(t=130, b=10, l=40, r=40),
            height=750,
        )

        fig.update_yaxes(
            title="Cumulative Return (%)", tickformat=".1%", fixedrange=True
        )
        fig.update_xaxes(
            title=dict(text="Date", font=dict(color="black")),
            tickfont=dict(color="black"),
            tickvals=custom_ticks,
            tickformat="%Y-%m-%d",
            hoverformat="%Y-%m-%d %H:%M",
            fixedrange=True,
        )

        fig.write_html(report_output_dir / chart_filename)

    if stats_df_dict:
        final_stats_df = pd.DataFrame(stats_df_dict)
        final_stats_df = final_stats_df.reindex(SELECTED_METRICS)

        formatted_df = final_stats_df.copy().astype(object)

        for metric in formatted_df.index:
            for col in formatted_df.columns:
                val = final_stats_df.loc[metric, col]
                if metric in FORMAT_MAP:
                    fmt = FORMAT_MAP[metric]
                    formatted_df.loc[metric, col] = (
                        fmt.format(val) if pd.notnull(val) else "-"
                    )
                else:
                    formatted_df.loc[metric, col] = (
                        "{:.2f}".format(val)
                        if isinstance(val, (int, float)) and pd.notnull(val)
                        else (val if pd.notnull(val) else "-")
                    )

        formatted_df = formatted_df.rename(index=RENAME_MAP)
        formatted_df.index.name = "Metrics"
        formatted_df = formatted_df.reset_index()

        main_table_html = formatted_df.to_html(
            classes="academic-table", border=0, index=False, justify="center"
        )
    else:
        main_table_html = "<p>No stats data available.</p>"

    css_style = """
        <style>
            body { margin: 0; padding: 20px; background-color: white; font-family: "Times New Roman", Times, serif; color: black; }
            .section-wrapper { width: 100%; margin-bottom: 40px; overflow: auto; text-align: center; }
            iframe { width: 100%; height: 800px; border: none; display: block; overflow: hidden; }

            .academic-table { width: 98%; margin: 30px auto; border-collapse: collapse; font-size: 11pt; table-layout: auto; }
            .academic-table th, .academic-table td { border: none; padding: 6px 8px; text-align: center; vertical-align: middle; white-space: nowrap; }

            .academic-table thead tr:first-child th { border-top: 2px solid black; border-bottom: 1px solid black; font-size: 11pt; padding-bottom: 8px; }
            .academic-table thead tr:nth-child(2) th { border-bottom: 1px solid black; font-style: italic; color: #333; padding-top: 6px; padding-bottom: 6px; font-size: 10pt; }
            .academic-table tbody tr:last-child td { border-bottom: 2px solid black; }

            .academic-table tbody td:first-child, .academic-table thead th:first-child { text-align: left; font-weight: bold; width: 220px; }
            .academic-table tbody tr:hover { background-color: #f9f9f9; }

            h3 { text-align: center; margin-bottom: 15px; font-size: 16pt; }
        </style>
    """

    full_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>Research Report {timestamp}</title>
        {css_style}
    </head>
    <body>
        <div class="section-wrapper">
            <iframe src="{chart_filename}" scrolling="no"></iframe>
        </div>
        <div class="section-wrapper">
            <h3>Performance Summary</h3>
            {main_table_html}
        </div>
    </body>
    </html>
    """

    final_path = report_output_dir / report_filename
    with open(final_path, "w", encoding="utf-8") as f:
        f.write(full_html)

    print(f"Report saved: {final_path}")


if __name__ == "__main__":
    generate_comparison_report(STRATEGY)
