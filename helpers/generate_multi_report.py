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
from runners.core.utils import generate_date_lists

logger = get_logger(__name__)

STRATEGIES = {
    "winner_rl_oos 1": "Winner 1",
    "winner_rl_oos 2 (time_delay_stop training)": "Winner 2",
}

LEVERAGE = 10
ELSEVIER_FONT = "Arial, sans-serif"
FONT_SERIF = "Times New Roman, serif"
FONT_SIZE_TICK = 10
FONT_SIZE_LABEL = 12
FONT_SIZE_TITLE = 13
COLOR_BLACK = "black"

PUBLICATION_COLORS = [
    "#1f77b4",
    "#d62728",
    "#2ca02c",
    "#9467bd",
    "#ff7f0e",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]

PDF_WIDTH = 720
PDF_HEIGHT = 400

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
    "avg_trade_duration",
    "sharpe_ratio_annual",
    "sortino_ratio_annual",
    "calmar_ratio",
]

RENAME_MAP = {
    "cagr": "CAGR",
    "volatility_annual": "Annual Volatility",
    "max_drawdown": "Max Drawdown",
    "win_count": "Win Count",
    "lose_count": "Loss Count",
    "win_rate": "Win Rate",
    "avg_win_return": "Avg Win",
    "avg_lose_return": "Avg Lose",
    "avg_trade_return": "Avg Trade Return",
    "avg_trade_duration": "Avg Trade Duration",
    "sharpe_ratio_annual": "Sharpe Ratio",
    "sortino_ratio_annual": "Sortino Ratio",
    "calmar_ratio": "Calmar Ratio",
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
    df_stats = (
        pd.read_parquet(stats_files[0]).set_index("metric") if stats_files else None
    )

    return df_returns, df_exec, df_stats


def get_run_config(base_dir: Path, strategy_name: str) -> dict:
    config_path = base_dir / strategy_name / ".hydra" / "config.yaml"
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}


def build_stitched_ewp(
    base_dir: Path, strategy_name: str, interval: Interval, assets_dict: dict
) -> pd.DataFrame:
    strat_dir = base_dir / strategy_name
    run_config = get_run_config(base_dir, strategy_name)
    if not run_config:
        logger.warning(f"Run config not found for {strategy_name}")
        return None

    config_dict = {
        "pair_selection_start": run_config["pair_selection"]["start"],
        "pair_selection_end": run_config["pair_selection"]["end"],
        "beta_test_start": run_config["performance"]["beta_start"],
        "test_start": run_config["performance"]["start"],
        "test_end": run_config["performance"]["end"],
    }
    lists = generate_date_lists(config_dict, run_config["performance"]["iterations"])

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

        ps_start = lists["pair_selection_start_list"][iter_num - 1]
        month_key = pd.to_datetime(ps_start).strftime("%Y-%m")

        month_data = assets_dict.get(month_key)
        iter_tickers = month_data.get("assets") if month_data else None

        if not iter_tickers:
            logger.warning(
                f"Warning: Lack of tickers for {month_key} (iter {iter_num}) in 'list_of_assets.json'"
            )
            continue

        fee_rate = float(run_config["market"]["fee_rate"])
        ewp_period = load_ewp_benchmark(
            tickers=iter_tickers,
            test_start=start_date,
            test_end=end_date,
            interval=interval,
            fee_rate=fee_rate,
        )

        col_name = "ewp_pct"
        ewp_dfs.append(ewp_period[[col_name]])

    if not ewp_dfs:
        return None

    final_ewp = pd.concat(ewp_dfs).sort_index()
    final_ewp = final_ewp[~final_ewp.index.duplicated(keep="first")]
    final_ewp["ewp_return"] = (1 + final_ewp["ewp_pct"]).cumprod() - 1

    return final_ewp


def generate_multi_report(strategies_map: dict):
    results_dir = project_root / "results"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_output_dir = results_dir / f"final_multi_report_{timestamp}"
    pdf_dir = report_output_dir / "pdfs"
    report_output_dir.mkdir(parents=True, exist_ok=True)
    pdf_dir.mkdir(parents=True, exist_ok=True)

    assets_file = project_root / "config" / "schemas" / "list_of_assets.json"
    list_of_assets = {}
    if assets_file.exists():
        with open(assets_file, "r", encoding="utf-8") as f:
            list_of_assets = json.load(f)

    axis_style_x = dict(
        showline=True,
        linewidth=1,
        linecolor=COLOR_BLACK,
        mirror=True,
        ticks="inside",
        tickcolor=COLOR_BLACK,
        tickwidth=1,
        title_standoff=5,
        tickfont=dict(family=ELSEVIER_FONT, size=FONT_SIZE_TICK, color=COLOR_BLACK),
        title_font=dict(family=ELSEVIER_FONT, size=FONT_SIZE_LABEL, color=COLOR_BLACK),
        showgrid=False,
    )

    axis_style_y = dict(
        showline=True,
        linewidth=1,
        linecolor=COLOR_BLACK,
        mirror=True,
        ticks="inside",
        tickcolor=COLOR_BLACK,
        tickwidth=1,
        title_standoff=5,
        tickfont=dict(family=ELSEVIER_FONT, size=FONT_SIZE_TICK, color=COLOR_BLACK),
        title_font=dict(family=ELSEVIER_FONT, size=FONT_SIZE_LABEL, color=COLOR_BLACK),
        showgrid=True,
        gridcolor="#E5E5E5",
        gridwidth=0.5,
        zeroline=True,
        zerolinecolor=COLOR_BLACK,
        zerolinewidth=1,
    )

    legend_style = dict(
        orientation="h",
        yanchor="top",
        y=-0.15,
        xanchor="center",
        x=0.5,
        font=dict(family=ELSEVIER_FONT, size=FONT_SIZE_TICK, color=COLOR_BLACK),
        bgcolor="rgba(255, 255, 255, 0)",
        borderwidth=0,
    )

    stats_dict = {}
    strategy_series = {}
    global_starts, global_ends = [], []

    first_strat = next(iter(strategies_map.keys()))
    config = get_run_config(results_dir, first_strat)
    initial_cash = float(config.get("market", {}).get("initial_cash", 100000))
    fee_rate = float(config.get("market", {}).get("fee_rate", 0.0005))
    risk_free_rate = float(config.get("market", {}).get("risk_free_rate_annual", 0.0))

    for i, (folder, label) in enumerate(strategies_map.items()):
        df_ret, df_exec, df_stats = load_strategy_data(results_dir, folder)
        if df_ret is None:
            continue

        global_starts.append(df_ret.index[0])
        global_ends.append(df_ret.index[-1])

        pnl = (df_ret["total_pnl"] - df_ret["total_fees"]) * LEVERAGE
        ret = pnl / initial_cash
        ret = ret - ret.iloc[0]

        df_lev = pd.DataFrame(index=df_ret.index)
        df_lev["total_pnl"] = pnl
        df_lev["total_net_pnl"] = pnl
        df_lev["equity"] = initial_cash + pnl

        strat_stats = calculate_stats(
            df_lev, df_exec, initial_cash, Interval.H1, risk_free_rate
        )
        stats_dict[f"{label} ({fee_rate*100}% fees, {LEVERAGE}x lev)"] = strat_stats[
            "net"
        ].reindex(SELECTED_METRICS)

        strategy_series[label] = {
            "series": ret,
            "color": PUBLICATION_COLORS[i % len(PUBLICATION_COLORS)],
            "label_full": f"{label} ({fee_rate*100}% fees, {LEVERAGE}x lev)",
        }

    if not global_starts:
        logger.error("No valid strategies loaded. Exiting.")
        return

    full_start = min(global_starts).strftime("%Y-%m-%d")
    full_end = max(global_ends).strftime("%Y-%m-%d")
    empty_ex = pd.DataFrame(columns=["position", "pnl", "fees", "entry_equity"])

    try:
        btc = load_btc_benchmark(full_start, full_end, Interval.H1, fee_rate)
        btc_ret = btc["BTC_return"] - btc["BTC_return"].iloc[0]

        df_btc = pd.DataFrame(index=btc_ret.index)
        df_btc["total_pnl"] = btc_ret * initial_cash
        df_btc["total_net_pnl"] = df_btc["total_pnl"]
        df_btc["equity"] = initial_cash + df_btc["total_net_pnl"]
        stats_dict[f"BTC B&H ({fee_rate*100}% fees)"] = calculate_stats(
            df_btc, empty_ex, initial_cash, Interval.H1, risk_free_rate
        )["net"].reindex(SELECTED_METRICS)
    except Exception as e:
        logger.error(f"Error during BTC data loading: {e}")
        btc_ret = None

    try:
        ewp_data = build_stitched_ewp(
            results_dir, first_strat, Interval.H1, list_of_assets
        )
        if ewp_data is not None:
            ewp_ret = ewp_data["ewp_return"] - ewp_data["ewp_return"].iloc[0]
            df_ewp = pd.DataFrame(index=ewp_ret.index)
            df_ewp["total_pnl"] = ewp_ret * initial_cash
            df_ewp["total_net_pnl"] = df_ewp["total_pnl"]
            df_ewp["equity"] = initial_cash + df_ewp["total_net_pnl"]
            stats_dict[f"EWP B&H ({fee_rate*100}% fees)"] = calculate_stats(
                df_ewp, empty_ex, initial_cash, Interval.H1, risk_free_rate
            )["net"].reindex(SELECTED_METRICS)
        else:
            ewp_ret = None
    except Exception as e:
        logger.error(f"Error during EWP data loading: {e}")
        ewp_ret = None

    plot_configs = [
        {"name": "strategies_only", "show_btc": False, "show_ewp": False},
        {"name": "with_btc", "show_btc": True, "show_ewp": False},
        {"name": "with_ewp", "show_btc": False, "show_ewp": True},
        {"name": "with_all", "show_btc": True, "show_ewp": True},
    ]

    final_fig_html = None

    for p_cfg in plot_configs:
        fig = go.Figure()

        for label, data in strategy_series.items():
            fig.add_trace(
                go.Scatter(
                    x=data["series"].index,
                    y=data["series"],
                    name=data["label_full"],
                    line=dict(color=data["color"], width=1.5, dash="solid"),
                )
            )

        if p_cfg["show_btc"] and btc_ret is not None:
            fig.add_trace(
                go.Scatter(
                    x=btc_ret.index,
                    y=btc_ret,
                    name=f"BTC B&H ({fee_rate*100}% fees)",
                    line=dict(color="gray", width=1.0, dash="dash"),
                )
            )

        if p_cfg["show_ewp"] and ewp_ret is not None:
            fig.add_trace(
                go.Scatter(
                    x=ewp_ret.index,
                    y=ewp_ret,
                    name=f"EWP B&H ({fee_rate*100}% fees)",
                    line=dict(color="black", width=1.0, dash="dot"),
                )
            )

        fig.update_layout(
            width=PDF_WIDTH,
            height=PDF_HEIGHT,
            font=dict(family=ELSEVIER_FONT, size=FONT_SIZE_LABEL, color=COLOR_BLACK),
            plot_bgcolor="white",
            paper_bgcolor="white",
            legend=legend_style,
            margin=dict(t=40, b=80, l=50, r=30),
            title=dict(
                text="Out-Of-Sample Performance (2025)",
                font=dict(size=FONT_SIZE_TITLE),
                x=0.5,
                xanchor="center",
                xref="paper",
            ),
        )

        if len(global_starts) > 0:
            first_series = list(strategy_series.values())[0]["series"]
            fig.update_xaxes(
                **axis_style_x,
                tickformat="%b\n%Y",
                dtick=(
                    "M1"
                    if (first_series.index[-1] - first_series.index[0]).days < 180
                    else "M3"
                ),
                tick0=first_series.index[0],
            )

        fig.update_yaxes(
            **axis_style_y, tickformat=".0%", title_text="Cumulative Return"
        )

        pdf_path = pdf_dir / f"comparison_{p_cfg['name']}.pdf"
        fig.write_image(str(pdf_path, format="pdf"))

        if p_cfg["name"] == "with_all":
            final_fig_html = fig

    df_stats = pd.DataFrame(stats_dict).rename(index=RENAME_MAP).astype(object)

    for col in df_stats.columns:
        for idx in df_stats.index:
            val = df_stats.loc[idx, col]
            if pd.notna(val):
                if idx in [
                    "CAGR",
                    "Annual Volatility",
                    "Max Drawdown",
                    "Win Rate",
                    "Avg Win",
                    "Avg Lose",
                    "Avg Trade Return",
                ]:
                    df_stats.loc[idx, col] = f"{val:.2%}"
                elif idx in ["Win Count", "Loss Count"]:
                    df_stats.loc[idx, col] = f"{int(val)}"
                else:
                    df_stats.loc[idx, col] = f"{val:.4f}"
            else:
                df_stats.loc[idx, col] = "-"

    with open(report_output_dir / "comparison_table.tex", "w") as f:
        f.write(
            df_stats.to_latex(
                column_format="l" + "r" * len(df_stats.columns), escape=False
            )
        )

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body {{ font-family: "{FONT_SERIF}"; padding: 30px; max-width: 1100px; margin: auto; background-color: #fcfcfc; }}
            .report-container {{ background-color: white; padding: 30px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }}
            h2, h3 {{ text-align: center; color: #333; }}
            .elsevier-table {{ width: 100%; border-collapse: collapse; margin-top: 20px; font-size: 10pt; }}
            .elsevier-table thead tr {{ border-top: 2px solid black; border-bottom: 1px solid black; }}
            .elsevier-table th, .elsevier-table td {{ padding: 8px; text-align: right; }}
            .elsevier-table td:first-child, .elsevier-table th:first-child {{ text-align: left; font-weight: bold; width: 200px; }}
            .elsevier-table tbody tr:last-child td {{ border-bottom: 2px solid black; }}
            .elsevier-table tbody tr:hover {{ background-color: #f5f5f5; }}
        </style>
    </head>
    <body>
        <div class="report-container">
            <h2>Multi-Strategy Performance Comparison</h2>
            <div>{final_fig_html.to_html(full_html=False, include_plotlyjs='cdn')}</div>
            <br>
            <h3>Table 1: Comparative Metrics</h3>
            {df_stats.to_html(classes="elsevier-table", border=0, justify="center")}
        </div>
    </body>
    </html>
    """
    with open(report_output_dir / "comparison_report.html", "w", encoding="utf-8") as f:
        f.write(html_content)

    logger.info(f"Comparison report generated in: {report_output_dir}")


if __name__ == "__main__":
    generate_multi_report(STRATEGIES)
