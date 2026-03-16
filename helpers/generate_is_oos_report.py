import json
import sys
import yaml
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

from modules.data_services.data_utils import load_btc_benchmark, load_ewp_benchmark
from modules.performance.stats import calculate_stats
from modules.core.enums import Interval
from modules.utils.logger import get_logger
from runners.core.utils import generate_date_lists

logger = get_logger(__name__)

STRATEGY = {
    "Rolling Beta-Hedge": {
        "IS": "winner_is",
        "OOS": "winner_oos",
    }
}

LEVERAGE = 10

ELSEVIER_FONT = "Arial, sans-serif"
FONT_SERIF = "Times New Roman, serif"
FONT_SIZE_TICK = 10
FONT_SIZE_LABEL = 12
FONT_SIZE_TITLE = 13
COLOR_BLACK = "black"

PDF_WIDTH = 720
PDF_HEIGHT = 350

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
    "avg_lose": "Avg Lose",
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
    base_dir: Path,
    strategy_name: str,
    interval: Interval,
    assets_dict: dict,
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

    col_name = "ewp_pct"
    final_ewp["ewp_return"] = (1 + final_ewp[col_name]).cumprod() - 1

    return final_ewp


def generate_academic_report(strategies_map: dict):
    results_dir = project_root / "results"

    report_output_dir = results_dir / "final_paper_is_oos"
    pdf_dir = report_output_dir / "pdfs"
    report_output_dir.mkdir(parents=True, exist_ok=True)
    pdf_dir.mkdir(parents=True, exist_ok=True)

    assets_file = project_root / "config" / "schemas" / "list_of_assets.json"
    list_of_assets = {}
    if assets_file.exists():
        with open(assets_file, "r", encoding="utf-8") as f:
            list_of_assets = json.load(f)
    else:
        logger.error(f"'list_of_assets.json' not found: {assets_file}")

    axis_style_x = dict(
        showline=True,
        linewidth=1,
        linecolor=COLOR_BLACK,
        mirror=True,
        ticks="inside",
        tickcolor=COLOR_BLACK,
        tickwidth=1,
        tickangle=0,
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
        y=-0.10,
        xanchor="center",
        x=0.5,
        font=dict(family=ELSEVIER_FONT, size=FONT_SIZE_TICK, color=COLOR_BLACK),
        bgcolor="rgba(255, 255, 255, 0)",
        bordercolor=COLOR_BLACK,
        borderwidth=0,
    )

    for label, folders in strategies_map.items():
        logger.info(f"Generating report for: {label}")

        run_config = get_run_config(results_dir, folders.get("OOS"))
        market_cfg = run_config.get("market", {})

        initial_cash = float(market_cfg.get("initial_cash"))
        fee_rate = float(market_cfg.get("fee_rate"))
        risk_free_rate = float(market_cfg.get("risk_free_rate_annual"))

        logger.info(
            f"Loaded config -> Initial Capital: {initial_cash}, Fee Rate: {fee_rate * 100}%, Risk Free Rate: {risk_free_rate}"
        )

        df_ret_is, df_exec_is, df_stats_is = load_strategy_data(
            results_dir, folders.get("IS")
        )
        df_ret_oos, df_exec_oos, df_stats_oos = load_strategy_data(
            results_dir, folders.get("OOS")
        )

        if df_ret_is is None or df_ret_oos is None:
            logger.warning(f"Data not found for IS/OOS ({label}). Skipping...")
            continue

        split_date = df_ret_oos.index[0]
        full_start = df_ret_is.index[0].strftime("%Y-%m-%d")
        full_end = df_ret_oos.index[-1].strftime("%Y-%m-%d")

        stats_dict = {}

        is_pnl = (df_ret_is["total_pnl"] - df_ret_is["total_fees"]) * LEVERAGE
        is_ret = is_pnl / initial_cash
        is_ret = is_ret - is_ret.iloc[0]

        df_is_lev = pd.DataFrame(index=df_ret_is.index)
        df_is_lev["total_pnl"] = is_pnl
        df_is_lev["total_net_pnl"] = is_pnl
        df_is_lev["equity"] = initial_cash + is_pnl

        stats_is_lev = calculate_stats(
            df_is_lev, df_exec_is, initial_cash, Interval.H1, risk_free_rate
        )
        stats_dict[f"Baseline ({fee_rate*100}% fees, {LEVERAGE}x leverage)"] = (
            stats_is_lev["net"].reindex(SELECTED_METRICS)
        )

        oos_pnl = (df_ret_oos["total_pnl"] - df_ret_oos["total_fees"]) * LEVERAGE
        oos_ret = oos_pnl / initial_cash
        oos_ret = oos_ret - oos_ret.iloc[0]

        df_oos_lev = pd.DataFrame(index=df_ret_oos.index)
        df_oos_lev["total_pnl"] = oos_pnl
        df_oos_lev["total_net_pnl"] = oos_pnl
        df_oos_lev["equity"] = initial_cash + oos_pnl

        stats_oos_lev = calculate_stats(
            df_oos_lev, df_exec_oos, initial_cash, Interval.H1, risk_free_rate
        )
        stats_dict[f"Baseline ({fee_rate*100}% fees, {LEVERAGE}x leverage)"] = (
            stats_oos_lev["net"].reindex(SELECTED_METRICS)
        )

        empty_ex = pd.DataFrame(columns=["position", "pnl", "fees", "entry_equity"])

        btc_is = load_btc_benchmark(
            full_start, split_date.strftime("%Y-%m-%d"), Interval.H1, fee_rate
        )
        btc_oos = load_btc_benchmark(
            split_date.strftime("%Y-%m-%d"), full_end, Interval.H1, fee_rate
        )
        btc_is_ret = btc_is["BTC_return"] - btc_is["BTC_return"].iloc[0]
        btc_oos_ret = btc_oos["BTC_return"] - btc_oos["BTC_return"].iloc[0]

        df_btc_oos = pd.DataFrame(index=btc_oos_ret.index)
        df_btc_oos["total_pnl"] = btc_oos_ret * initial_cash
        df_btc_oos["total_net_pnl"] = df_btc_oos["total_pnl"]
        df_btc_oos["equity"] = initial_cash + df_btc_oos["total_net_pnl"]
        stats_dict[f"BTC B&H ({fee_rate*100}% fees)"] = calculate_stats(
            df_btc_oos, empty_ex, initial_cash, Interval.H1, risk_free_rate
        )["net"].reindex(SELECTED_METRICS)

        # EWP
        ewp_data_is = build_stitched_ewp(
            results_dir, folders.get("IS"), Interval.H1, list_of_assets
        )
        ewp_data_oos = build_stitched_ewp(
            results_dir, folders.get("OOS"), Interval.H1, list_of_assets
        )
        ewp_is_ret = ewp_data_is["ewp_return"] - ewp_data_is["ewp_return"].iloc[0]
        ewp_oos_ret = ewp_data_oos["ewp_return"] - ewp_data_oos["ewp_return"].iloc[0]

        df_ewp_oos = pd.DataFrame(index=ewp_oos_ret.index)
        df_ewp_oos["total_pnl"] = ewp_oos_ret * initial_cash
        df_ewp_oos["total_net_pnl"] = df_ewp_oos["total_pnl"]
        df_ewp_oos["equity"] = initial_cash + df_ewp_oos["total_net_pnl"]
        stats_dict[f"EWP B&H ({fee_rate*100}% fees)"] = calculate_stats(
            df_ewp_oos, empty_ex, initial_cash, Interval.H1, risk_free_rate
        )["net"].reindex(SELECTED_METRICS)

        plot_configs = [
            {"name": "baseline_only", "show_btc": False, "show_ewp": False},
            {"name": "with_btc", "show_btc": True, "show_ewp": False},
            {"name": "with_ewp", "show_btc": False, "show_ewp": True},
            {"name": "with_all", "show_btc": True, "show_ewp": True},
        ]

        final_fig_html = None

        for p_cfg in plot_configs:
            fig = make_subplots(
                rows=1,
                cols=2,
                shared_yaxes=True,
                horizontal_spacing=0.03,
                subplot_titles=[
                    "Panel A: In-Sample Performance (2024)",
                    "Panel B: Out-of-Sample Performance (2025)",
                ],
            )

            for c, (ret, name) in enumerate([(is_ret, "IS"), (oos_ret, "OOS")], 1):
                fig.add_trace(
                    go.Scatter(
                        x=ret.index,
                        y=ret,
                        name=f"Baseline ({fee_rate*100}% fees, {LEVERAGE}x leverage)",
                        legendgroup="M",
                        line=dict(color=COLOR_BLACK, width=1.5),
                        showlegend=(c == 1),
                    ),
                    row=1,
                    col=c,
                )

            if p_cfg["show_btc"]:
                for c, ret in enumerate([btc_is_ret, btc_oos_ret], 1):
                    fig.add_trace(
                        go.Scatter(
                            x=ret.index,
                            y=ret,
                            name=f"BTC B&H ({fee_rate*100}% fees)",
                            legendgroup="B",
                            line=dict(color=COLOR_BLACK, width=1.0, dash="dash"),
                            showlegend=(c == 1),
                        ),
                        row=1,
                        col=c,
                    )

            if p_cfg["show_ewp"]:
                for c, ret in enumerate([ewp_is_ret, ewp_oos_ret], 1):
                    fig.add_trace(
                        go.Scatter(
                            x=ret.index,
                            y=ret,
                            name=f"EWP B&H ({fee_rate*100}% fees)",
                            legendgroup="E",
                            line=dict(color=COLOR_BLACK, width=1.0, dash="dot"),
                            showlegend=(c == 1),
                        ),
                        row=1,
                        col=c,
                    )

            fig.update_layout(
                width=PDF_WIDTH,
                height=PDF_HEIGHT,
                font=dict(
                    family=ELSEVIER_FONT, size=FONT_SIZE_LABEL, color=COLOR_BLACK
                ),
                plot_bgcolor="white",
                paper_bgcolor="white",
                showlegend=True,
                legend=legend_style,
                margin=dict(t=30, b=40, l=45, r=10),
            )

            for annotation in fig["layout"]["annotations"]:
                annotation["font"] = dict(
                    family=ELSEVIER_FONT, size=FONT_SIZE_TITLE, color=COLOR_BLACK
                )
                annotation["yshift"] = 5

            fig.update_xaxes(
                **axis_style_x,
                tickformat="%b\n%Y",
                dtick="M3",
                tick0=is_ret.index[0] if len(is_ret) > 0 else None,
            )

            fig.update_yaxes(**axis_style_y, tickformat=".0%")

            fig.update_yaxes(title_text="Cumulative Return", row=1, col=1)

            pdf_path = pdf_dir / f"{label}_equity_{p_cfg['name']}.pdf"
            fig.write_image(str(pdf_path), format="pdf")

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
                    ]:
                        df_stats.loc[idx, col] = f"{val:.2%}"
                    elif idx in ["Win Count", "Loss Count"]:
                        df_stats.loc[idx, col] = f"{int(val)}"
                    else:
                        df_stats.loc[idx, col] = f"{val:.4f}"
                else:
                    df_stats.loc[idx, col] = "-"

        with open(report_output_dir / f"{label}_table.tex", "w") as f:
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
                body {{ font-family: "{FONT_SERIF}"; padding: 30px; max-width: 1200px; margin: auto; background-color: #fcfcfc; }}
                .report-container {{ background-color: white; padding: 30px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }}
                h2, h3 {{ text-align: center; color: #333; }}
                .elsevier-table {{ width: 100%; border-collapse: collapse; margin-top: 20px; font-size: 11pt; }}
                .elsevier-table thead tr {{ border-top: 2px solid black; border-bottom: 1px solid black; }}
                .elsevier-table th, .elsevier-table td {{ padding: 8px; text-align: right; }}
                .elsevier-table td:first-child, .elsevier-table th:first-child {{ text-align: left; font-weight: bold; width: 200px; }}
                .elsevier-table tbody tr:last-child td {{ border-bottom: 2px solid black; }}
                .elsevier-table tbody tr:hover {{ background-color: #f5f5f5; }}
            </style>
        </head>
        <body>
            <div class="report-container">
                <h2>{label} - In-Sample vs Out-of-Sample</h2>
                <div>{final_fig_html.to_html(full_html=False, include_plotlyjs='cdn')}</div>
                <br><br>
                <h3>Table 1: Performance Summary</h3>
                {df_stats.to_html(classes="elsevier-table", border=0, justify="center")}
            </div>
        </body>
        </html>
        """
        with open(
            report_output_dir / f"{label}_report.html", "w", encoding="utf-8"
        ) as f:
            f.write(html_content)
        logger.info("IS-OOS Pipeline completed successfully.")


if __name__ == "__main__":
    generate_academic_report(STRATEGY)
