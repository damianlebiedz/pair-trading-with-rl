import sys
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
from datetime import datetime

from modules.utils.plots import _get_custom_tickvals
from modules.data_services.data_utils import load_btc_benchmark

# ==========================================
STRATEGIES = {
    "WINNER_1_baseline": "Fixed | No Hedge",
    "WINNER_5_hybrid_fixed": "Fixed | Rolling Beta-Hedge",
}
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

    stats_files = list(strat_dir.glob("stats_*.parquet"))
    df_stats = pd.read_parquet(stats_files[0]) if stats_files else None

    return df_returns, df_stats


def generate_comparison_report(strategies_input: dict | list) -> None:
    results_dir = project_root / "results"
    report_output_dir = results_dir / "report"
    report_output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    chart_filename = f"comparison_chart_{timestamp}.html"
    report_filename = f"report_{timestamp}.html"

    strategies_returns = {}
    stats_df_dict = {}

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

                # Rozbicie na MultiIndex: Strategia -> Gross/Net
                if "gross" in df_stat.columns:
                    stats_df_dict[(label, "Gross")] = df_stat["gross"].reindex(
                        SELECTED_METRICS
                    )
                if "net" in df_stat.columns:
                    stats_df_dict[(label, "Net")] = df_stat["net"].reindex(
                        SELECTED_METRICS
                    )

            except Exception as e:
                print(f"[ERROR] Processing stats for {label}: {e}")

    if strategies_returns:
        global_start = min(all_dates).strftime("%Y-%m-%d")
        global_end = max(all_dates).strftime("%Y-%m-%d")

        btc_data = load_btc_benchmark(
            test_start=global_start, test_end=global_end, interval="1h"
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
        first_df = list(strategies_returns.values())[0]
        custom_ticks = (
            _get_custom_tickvals(first_df.index) if hasattr(first_df, "index") else []
        )

        color_idx = 0
        for name, df in strategies_returns.items():

            if "total_net_return" in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df["total_net_return"],
                        mode="lines",
                        name=f"{name} (Net)",
                        line=dict(color=colors[color_idx % len(colors)], width=2),
                        hovertemplate=f"<b>{name} (Net)</b>: %{{y:.2%}}<extra></extra>",
                    )
                )

            if "total_return" in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df["total_return"],
                        mode="lines",
                        name=f"{name} (Gross)",
                        line=dict(
                            color=colors[color_idx % len(colors)], width=2, dash="dot"
                        ),
                        hovertemplate=f"<b>{name} (Gross)</b>: %{{y:.2%}}<extra></extra>",
                        visible="legendonly",
                    )
                )

            color_idx += 1

        if btc_data is not None and not btc_data.empty:
            if btc_data.index.tz is not None and first_df.index.tz is None:
                btc_data.index = btc_data.index.tz_localize(None)
            elif btc_data.index.tz is None and first_df.index.tz is not None:
                btc_data.index = btc_data.index.tz_localize(first_df.index.tz)

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
                            line=dict(color="grey", width=1.5, dash="dash"),
                            opacity=0.6,
                            hovertemplate="<b>BTC</b>: %{{y:.2%}}<extra></extra>",
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
            hovermode="x unified",
            legend=dict(orientation="h", y=1.05, x=0.5, xanchor="center"),
            margin=dict(t=80),
            height=800,
        )

        fig.update_yaxes(
            title="Cumulative Return (%)", tickformat=".1%", fixedrange=True
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
        final_stats_df.index = final_stats_df.index.astype(str)
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

            /* --- STYL AKADEMICKI (LaTeX Booktabs) --- */
            .academic-table {
                width: 85%;
                margin: 30px auto;
                border-collapse: collapse; 
                font-size: 12pt;
                table-layout: fixed; /* Wymusza równe szerokości kolumn */
            }

            .academic-table th,
            .academic-table td {
                border: none; /* Całkowity brak pionowych kresek */
                padding: 10px 12px;
                text-align: center;
                vertical-align: middle;
            }

            /* Główna gruba linia na samej górze tabeli */
            .academic-table thead tr:first-child th {
                border-top: 2px solid black;
                border-bottom: 1px solid black; /* Oddziela nazwy strategii od Gross/Net */
                font-size: 13pt;
                padding-bottom: 10px;
            }

            /* Linia pod Gross/Net zamykająca nagłówek */
            .academic-table thead tr:nth-child(2) th {
                border-bottom: 1px solid black;
                font-style: italic;
                color: #333;
                padding-top: 8px;
                padding-bottom: 8px;
            }

            /* Główna gruba linia na samym dole tabeli */
            .academic-table tbody tr:last-child td {
                border-bottom: 2px solid black;
            }

            /* Wyrównanie pierwszej kolumny (Metryki) do lewej i ustawienie jej szerokości */
            .academic-table tbody td:first-child,
            .academic-table thead th:first-child {
                text-align: left;
                width: 25%;
                font-weight: bold;
            }

            /* Delikatny efekt najechania myszką ułatwiający czytanie wierszy */
            .academic-table tbody tr:hover {
                background-color: #f9f9f9;
            }

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
    generate_comparison_report(STRATEGIES)
