import sys
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
from datetime import datetime

from modules.utils.plots import _get_custom_tickvals
from modules.data_services.data_utils import load_btc_benchmark


current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
sys.path.append(str(project_root))

SELECTED_METRICS = [
    "total_return",
    "cagr",
    "volatility_annual",
    "max_drawdown",
    "win_count",
    "lose_count",
    "win_rate",
    "max_win",
    "max_lose",
    "avg_win_return",
    "avg_lose_return",
    "avg_trade_return",
    "sharpe_ratio_annual",
    "sortino_ratio_annual",
    "calmar_ratio_annual",
]

FORMAT_MAP = {
    "total_return": "{:.2%}",
    "cagr": "{:.2%}",
    "volatility_annual": "{:.2%}",
    "max_drawdown": "{:.2%}",
    "win_rate": "{:.2%}",
    "max_win": "{:.2%}",
    "max_lose": "{:.2%}",
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
    "total_return": "Total Return",
    "cagr": "CAGR",
    "volatility_annual": "Annual Volatility",
    "max_drawdown": "Max Drawdown",
    "win_count": "Win Count",
    "lose_count": "Lose Count",
    "win_rate": "Win Rate",
    "max_win": "Max Win",
    "max_lose": "Max Lose",
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

    stats_files = list(strat_dir.glob("stats_*.parquet"))
    df_stats = pd.read_parquet(stats_files[0]) if stats_files else None

    return df_returns, df_stats


def load_pair_selections(base_dir: Path, strategy_name: str) -> pd.Series:
    strat_dir = base_dir / strategy_name
    counts = {}

    iteration = 1
    while (strat_dir / str(iteration)).exists():
        iter_path = strat_dir / str(iteration)
        files = list(iter_path.glob("pair_selection_*.parquet"))

        if files:
            df = pd.read_parquet(files[0])
            counts[iteration] = len(df)
        else:
            counts[iteration] = 0

        iteration += 1

    if not counts:
        files = list(strat_dir.glob("pair_selection_*.parquet"))
        if files:
            df = pd.read_parquet(files[0])
            counts[1] = len(df)

    return pd.Series(counts, dtype="int")


def generate_comparison_report(strategies_input: dict | list) -> None:
    results_dir = project_root / "results"
    report_output_dir = results_dir / "report"
    report_output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    chart_filename = f"comparison_chart_{timestamp}.html"
    report_filename = f"report_{timestamp}.html"

    strategies_returns = {}
    stats_df = {}

    all_dates = []

    if isinstance(strategies_input, list):
        strategies_map = {s: s for s in strategies_input}
    else:
        strategies_map = strategies_input

    for folder, label in strategies_map.items():
        print(f"Loading: {folder}...")
        df_ret, df_stat_raw = load_strategy_data(results_dir, folder)

        if df_ret is not None:
            strategies_returns[label] = df_ret
            if not df_ret.empty:
                all_dates.extend(df_ret.index.tolist())

        if df_stat_raw is not None:
            try:
                df_stat = df_stat_raw.copy()

                if "metric" not in df_stat.columns:
                    raise ValueError("stats.parquet must contain 'metric' column")

                df_stat["metric"] = df_stat["metric"].astype(str)
                df_stat = df_stat.set_index("metric")

                if "net" in df_stat.columns:
                    s = df_stat["net"]
                else:
                    s = df_stat.iloc[:, 0]

                s = s.reindex(SELECTED_METRICS)
                stats_df[label] = s

            except Exception as e:
                print(f"[ERROR] Processing stats for {label}: {e}")

    if strategies_returns:
        global_start = min(all_dates).strftime("%Y-%m-%d")
        global_end = max(all_dates).strftime("%Y-%m-%d")

        btc_data = load_btc_benchmark(
            test_start=global_start, test_end=global_end, interval="1h"
        )

        fig = go.Figure()
        colors = ["#1f77b4", "#2ca02c", "#d62728", "#9467bd", "#ff7f0e"]
        first_df = list(strategies_returns.values())[0]
        custom_ticks = (
            _get_custom_tickvals(first_df.index) if hasattr(first_df, "index") else []
        )

        color_idx = 0
        for name, df in strategies_returns.items():
            y_col = (
                "net_return_pct" if "net_return_pct" in df.columns else df.columns[0]
            )
            y_col_gross = (
                "total_return_pct" if "total_return_pct" in df.columns else None
            )

            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df[y_col],
                    mode="lines",
                    name=f"{name} (Net)",
                    line=dict(color=colors[color_idx % len(colors)], width=2),
                    hovertemplate=f"<b>Date</b>: %{{x|%Y-%m-%d %H:%M}}<br><b>{name}</b>: %{{y:.2%}}<extra></extra>",
                )
            )

            if y_col_gross:
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df[y_col_gross],
                        mode="lines",
                        name=f"{name} (Gross)",
                        line=dict(
                            color=colors[color_idx % len(colors)], width=2, dash="dot"
                        ),
                        hovertemplate=f"<b>Date</b>: %{{x|%Y-%m-%d %H:%M}}<br><b>{name}</b>: %{{y:.2%}}<extra></extra>",
                        visible="legendonly",
                    )
                )
            color_idx += 1

        if btc_data is not None and not btc_data.empty:
            btc_sub = btc_data.loc[global_start:global_end].copy()

            if not btc_sub.empty:
                col = "close" if "close" in btc_sub.columns else btc_sub.columns[0]
                btc_ret = (btc_sub[col] / btc_sub[col].iloc[0]) - 1

                fig.add_trace(
                    go.Scatter(
                        x=btc_sub.index,
                        y=btc_ret,
                        mode="lines",
                        name="BTC Benchmark",
                        line=dict(color="grey", width=1.5, dash="dot"),
                        opacity=0.6,
                        hovertemplate="<b>Date</b>: %{{x|%Y-%m-%d %H:%M}}<br><b>BTC</b>: %{{y:.2%}}<extra></extra>",
                        visible="legendonly",
                    )
                )

        fig.update_layout(
            title=dict(
                text="Comparison of Multi-Pair Strategies",
                x=0.5,
                font=dict(color="black", size=20),
            ),
            template="plotly_white",
            hovermode="closest",
            legend=dict(orientation="h", y=1.05, x=0.5, xanchor="center"),
            margin=dict(t=80),
            height=800,
        )

        fig.update_yaxes(title="Total Return (%)", tickformat=".1%", fixedrange=True)
        fig.update_xaxes(
            title=dict(text="Date", font=dict(color="black")),
            tickfont=dict(color="black"),
            tickvals=custom_ticks,
            tickformat="%Y-%m-%d",
            fixedrange=True,
        )

        fig.write_html(report_output_dir / chart_filename)

    if stats_df:
        final_stats_df = pd.DataFrame(stats_df)
        final_stats_df.index = final_stats_df.index.astype(str)
        final_stats_df = final_stats_df.reindex(SELECTED_METRICS)
        final_stats_df = final_stats_df.drop(index="metric", errors="ignore")

        formatted_df = final_stats_df.copy().astype(object)

        for metric in formatted_df.index:
            if metric in FORMAT_MAP:
                fmt = FORMAT_MAP[metric]
                formatted_df.loc[metric] = final_stats_df.loc[metric].apply(
                    lambda x: fmt.format(x) if pd.notnull(x) else "-"
                )
            else:
                formatted_df.loc[metric] = final_stats_df.loc[metric].apply(
                    lambda x: "{:.2f}".format(x) if isinstance(x, (int, float)) else x
                )

        formatted_df = formatted_df.rename(index=RENAME_MAP)

        formatted_df = formatted_df.reset_index()
        formatted_df = formatted_df.rename(columns={"index": ""})

        main_table_html = formatted_df.to_html(
            classes="academic-table", border=0, index=False
        )
    else:
        main_table_html = "<p>No stats data available.</p>"

    df_all_counts = pd.DataFrame()
    for folder, label in strategies_map.items():
        s_counts = load_pair_selections(results_dir, folder)
        if not s_counts.empty:
            df_all_counts[label] = s_counts

    if not df_all_counts.empty:
        df_all_counts = df_all_counts.sort_index()
        df_all_counts = df_all_counts.fillna(0).astype(int)
        df_all_counts.index.name = None

        tbl_html = df_all_counts.to_html(classes="academic-table pair-table", border=0)
        pair_selection_section = f"<h3>Pair Counts Summary</h3>{tbl_html}"
    else:
        pair_selection_section = "<p>No pair selection data.</p>"

    css_style = """
    <style>
        body {
            margin: 0;
            padding: 20px;
            background-color: white;
            font-family: "Times New Roman", Times, serif;
            color: black;
        }

        .section-wrapper {
            width: 100%;
            margin-bottom: 40px;
            overflow: hidden;
            text-align: center;
        }

        iframe {
            width: 100%;
            height: 800px;
            border: none;
            display: block;
            overflow: hidden;
        }

        .academic-table {
            width: 85%;
            margin: 20px auto;
            border-collapse: collapse; 
            font-size: 14pt;
            table-layout: fixed;
        }

        .academic-table th,
        .academic-table td {
            border: 1px solid black;
            padding: 10px 12px;
            text-align: center;
        }

        .academic-table tbody th {
            text-align: center;
            font-weight: bold;
            border: 1px solid black;
        }

        .pair-table {
            width: 50%;
        }
        .pair-table th {
            background-color: #f9f9f9;
        }

        h3 { text-align: center; margin-bottom: 15px; }
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

        <div class="section-wrapper">
            {pair_selection_section}
        </div>

    </body>
    </html>
    """

    final_path = report_output_dir / report_filename
    with open(final_path, "w", encoding="utf-8") as f:
        f.write(full_html)

    print(f"Report saved: {final_path}")


if __name__ == "__main__":
    my_strategies = {
        "2026-01-25_16-23-31": "Static (2/0)",
        "2026-01-25_16-26-41": "Rolling (2/0)",
        # "...": "Static (Opt, Both)",
        # "...": "Rolling (Opt, Both)",
        # "...": "Static (Opt, 2nd)",
        # "...": "Rolling (Opt, 2nd)",
    }
    generate_comparison_report(my_strategies)
