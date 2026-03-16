import yaml
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
from plotly.subplots import make_subplots
from collections import defaultdict

from modules.core.enums import Interval
from modules.performance.stats import calculate_stats
from modules.utils.logger import get_logger

logger = get_logger(__name__)

FOLDER = "sa av"

IS_BASE_DIR = "is"
OOS_BASE_DIR = "oos"

BASELINE = {
    "IS": "winner_is",
    "OOS": "winner_oos",
}

LEVERAGE = 10.0

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
PDF_HEIGHT = 350

SENSITIVITY_PARAMS = [
    "entry_threshold",
    "exit_threshold",
    "stop_loss",
    "top_n",
    "z_score_window",
]

ASSUMPTIONS = [
    "fee_rate",
    "beta_hedge",
    "sl_lock",
    "time_decay_sl",
]

SELECTED_METRICS = [
    "CAGR",
    "Annual Volatility",
    "Max Drawdown",
    "Win Count",
    "Lose Count",
    "Win Rate",
    "Avg Win",
    "Avg Lose",
    "Avg Trade Return",
    "Avg Trade Duration",
    "Sharpe Ratio",
    "Sortino Ratio",
    "Calmar Ratio",
]

NAME_MAP = {
    "entry_threshold": "Entry Threshold",
    "exit_threshold": "Exit Threshold",
    "stop_loss": "Stop Loss",
    "top_n": "Pairs",
    "z_score_window": "Z-Score Window",
    "fee_rate": "Fee Rate",
    "beta_hedge": "Beta Hedge",
    "sl_lock": "SL Lock",
    "time_decay_sl": "Time Decay SL",
}

FORMAT_MAP = {
    "CAGR": "{:.2%}",
    "Annual Volatility": "{:.2%}",
    "Max Drawdown": "{:.2%}",
    "Win Count": "{:.0f}",
    "Lose Count": "{:.0f}",
    "Win Rate": "{:.2%}",
    "Avg Win": "{:.2%}",
    "Avg Lose": "{:.2%}",
    "Avg Trade Return": "{:.2%}",
    "Avg Trade Duration": "{:.2f}",
    "Sharpe Ratio": "{:.4f}",
    "Sortino Ratio": "{:.4f}",
    "Calmar Ratio": "{:.4f}",
}

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
    y=-0.15,
    xanchor="center",
    x=0.5,
    font=dict(family=ELSEVIER_FONT, size=FONT_SIZE_TICK, color=COLOR_BLACK),
    bgcolor="rgba(255, 255, 255, 0)",
    bordercolor=COLOR_BLACK,
    borderwidth=0,
)


def generate_reports(folder_name: str, baseline_dict: dict):
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    category_dir = project_root / "results" / folder_name

    is_base_dir = category_dir / "is"
    oos_base_dir = category_dir / "oos"

    paper_out_dir = category_dir / "final_paper_reports"
    pdf_dir = paper_out_dir / "pdfs"
    paper_out_dir.mkdir(parents=True, exist_ok=True)
    pdf_dir.mkdir(parents=True, exist_ok=True)

    if not is_base_dir.exists() or not oos_base_dir.exists():
        raise ValueError(f"Directories 'is' and 'oos' must exist inside {category_dir}")

    base_config_path = project_root / "config" / "base.yaml"
    if base_config_path.exists():
        with open(base_config_path, "r", encoding="utf-8") as f:
            base_config = yaml.safe_load(f)
            market_cfg = base_config.get("market", {})
            initial_cash = market_cfg.get("initial_cash")
    else:
        raise ValueError("'base.yaml' not found")

    def get_run_data(run_dir: Path):
        config_path = run_dir / ".hydra" / "config.yaml"
        ts_files = list(run_dir.glob("returns_multi_pair_*.parquet"))
        exec_files = list(run_dir.glob("exec_logger_*.parquet"))

        if not (config_path.exists() and ts_files and exec_files):
            return None

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)

            ts_df = pd.read_parquet(ts_files[0])
            exec_df = pd.read_parquet(exec_files[0])

            def get_cfg(key):
                if key in config:
                    return config[key]
                for section in ["performance", "pair_selection", "market", "settings"]:
                    if section in config and key in config[section]:
                        return config[section][key]
                return None

            params = {k: get_cfg(k) for k in (SENSITIVITY_PARAMS + ASSUMPTIONS)}

            lev_pnl = (ts_df["total_pnl"] - ts_df["total_fees"]) * LEVERAGE
            ret_series = lev_pnl / initial_cash

            df_lev = pd.DataFrame(index=ts_df.index)
            df_lev["total_pnl"] = lev_pnl
            df_lev["total_net_pnl"] = lev_pnl
            df_lev["equity"] = initial_cash + lev_pnl

            risk_free_rate = float(
                config.get("market", {}).get("risk_free_rate_annual", 0.0)
            )

            stats_lev = calculate_stats(
                df_lev, exec_df, initial_cash, Interval.H1, risk_free_rate
            )
            net_stats = stats_lev["net"]

            metrics = {
                "CAGR": net_stats.get("cagr"),
                "Annual Volatility": net_stats.get("volatility_annual"),
                "Max Drawdown": net_stats.get("max_drawdown"),
                "Win Count": net_stats.get("win_count"),
                "Lose Count": net_stats.get("lose_count"),
                "Win Rate": net_stats.get("win_rate"),
                "Avg Win": net_stats.get("avg_win_return"),
                "Avg Lose": net_stats.get("avg_lose_return"),
                "Avg Trade Return": net_stats.get("avg_trade_return"),
                "Avg Trade Duration": net_stats.get("avg_trade_duration"),
                "Sharpe Ratio": net_stats.get("sharpe_ratio_annual"),
                "Sortino Ratio": net_stats.get("sortino_ratio_annual"),
                "Calmar Ratio": net_stats.get("calmar_ratio"),
            }

            return {
                "params": params,
                "metrics": metrics,
                "ret_series": ret_series,
                "id": run_dir.name,
            }
        except Exception as e:
            logger.error(f"Error during loading {run_dir.name}: {str(e)}")
            return None

    base_is = get_run_data(is_base_dir / baseline_dict["IS"])
    base_oos = get_run_data(oos_base_dir / baseline_dict["OOS"])

    if not base_is or not base_oos:
        raise ValueError(
            "Error loading Baseline (IS or OOS). Make sure the winner folders exist."
        )

    base_is_series = base_is["ret_series"] - base_is["ret_series"].iloc[0]
    base_oos_series = base_oos["ret_series"] - base_oos["ret_series"].iloc[0]

    sensitivity_results, assumptions_results = [], []
    sensitivity_plots, mechanism_plots = [], []

    base_entry = {
        **base_oos["params"],
        **base_oos["metrics"],
        "Variation": "Baseline",
        "Group": "Baseline",
    }
    sensitivity_results.append(base_entry)
    assumptions_results.append(base_entry)

    def is_different(val1, val2):
        s1, s2 = str(val1).lower().strip(), str(val2).lower().strip()
        if s1 == s2:
            return False
        try:
            return float(val1) != float(val2)
        except (ValueError, TypeError):
            return True

    def params_to_key(p):
        return tuple(sorted(p.items()))

    is_runs_dict = {}
    for d in is_base_dir.iterdir():
        if d.is_dir() and d.name != baseline_dict["IS"]:
            run_data = get_run_data(d)
            if run_data:
                key = params_to_key(run_data["params"])
                is_runs_dict[key] = run_data

    for oos_run_dir in oos_base_dir.iterdir():
        if not oos_run_dir.is_dir() or oos_run_dir.name == baseline_dict["OOS"]:
            continue

        run_oos = get_run_data(oos_run_dir)
        if not run_oos:
            continue

        key = params_to_key(run_oos["params"])
        run_is = is_runs_dict.get(key)

        if not run_is:
            logger.warning(
                f"Variation OOS {oos_run_dir.name} has parameters, which are not found in IS. Skipping."
            )
            continue

        diffs = [
            k
            for k in (SENSITIVITY_PARAMS + ASSUMPTIONS)
            if is_different(run_oos["params"][k], base_oos["params"][k])
        ]

        if len(diffs) == 1:
            p_name = diffs[0]
            val = run_oos["params"][p_name]

            if p_name == "beta_hedge" and val == "no_hedge":
                val = False
            elif p_name == "fee_rate":
                val = f"{float(val):.2%}"

            var_label = f"{NAME_MAP.get(p_name, p_name)} = {val}"

            is_series = run_is["ret_series"] - run_is["ret_series"].iloc[0]
            oos_series = run_oos["ret_series"] - run_oos["ret_series"].iloc[0]

            plot_item = {
                "series_is": is_series,
                "series_oos": oos_series,
                "name": var_label,
                "group": p_name,
            }

            entry = {
                **run_oos["params"],
                **run_oos["metrics"],
                "Variation": var_label,
                "Group": p_name,
            }

            if p_name in SENSITIVITY_PARAMS:
                sensitivity_plots.append(plot_item)
                sensitivity_results.append(entry)
            else:
                mechanism_plots.append(plot_item)
                assumptions_results.append(entry)

    def save_combined_report(results, plots, prefix, title):
        if len(results) <= 1:
            logger.info(f"Skipping '{title}' (only baseline found).")
            return

        df_raw = pd.DataFrame(results)

        group_order = ["Baseline"] + SENSITIVITY_PARAMS + ASSUMPTIONS
        df_raw["Group"] = pd.Categorical(
            df_raw["Group"], categories=group_order, ordered=True
        )
        df_raw = df_raw.sort_values(by=["Group", "Variation"])
        df_raw.to_parquet(paper_out_dir / f"{prefix}_stats.parquet", index=False)

        plots_by_group = defaultdict(list)
        for p in plots:
            plots_by_group[p["group"]].append(p)

        html_plots = ""

        for group_name, group_plots in plots_by_group.items():
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

            for c, (ret, name) in enumerate(
                [(base_is_series, "IS"), (base_oos_series, "OOS")], 1
            ):
                fig.add_trace(
                    go.Scatter(
                        x=ret.index,
                        y=ret,
                        name="Baseline",
                        legendgroup="Baseline",
                        line=dict(color=COLOR_BLACK, width=2.0),
                        showlegend=(c == 1),
                    ),
                    row=1,
                    col=c,
                )

            for i, p in enumerate(group_plots):
                color = PUBLICATION_COLORS[i % len(PUBLICATION_COLORS)]

                for c, ret in enumerate([p["series_is"], p["series_oos"]], 1):
                    fig.add_trace(
                        go.Scatter(
                            x=ret.index,
                            y=ret,
                            name=p["name"],
                            legendgroup=p["name"],
                            line=dict(color=color, width=1.5, dash="solid"),
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
                margin=dict(t=30, b=50, l=45, r=10),
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
                tick0=base_is_series.index[0] if len(base_is_series) > 0 else None,
                row=1,
                col=1,
            )

            fig.update_xaxes(
                **axis_style_x,
                tickformat="%b\n%Y",
                dtick="M3",
                tick0=base_oos_series.index[0] if len(base_oos_series) > 0 else None,
                row=1,
                col=2,
            )

            fig.update_yaxes(**axis_style_y, tickformat=".0%")
            fig.update_yaxes(title_text="Cumulative Return", row=1, col=1)

            pdf_path = pdf_dir / f"{prefix}_{group_name}.pdf"
            try:
                fig.write_image(str(pdf_path), format="pdf")
                logger.info(f"PDF saved: {pdf_path}")
            except Exception as e:
                logger.error(f"Error during PDF export: {e}")

            html_plots += f"<div style='margin-bottom: 40px;'>{fig.to_html(full_html=False, include_plotlyjs='cdn')}</div>"

        df_tab = df_raw.set_index("Variation")[SELECTED_METRICS]
        formatted_df = df_tab.copy().astype(object)

        for variation in formatted_df.index:
            for metric in formatted_df.columns:
                val = df_tab.loc[variation, metric]
                if pd.notnull(val):
                    formatted_df.loc[variation, metric] = FORMAT_MAP.get(
                        metric, "{:.2f}"
                    ).format(val)
                else:
                    formatted_df.loc[variation, metric] = "-"

        with open(paper_out_dir / f"{prefix}_table.tex", "w") as f:
            f.write(
                formatted_df.to_latex(
                    column_format="l" + "r" * len(formatted_df.columns), escape=False
                )
            )

        main_table_html = formatted_df.reset_index().to_html(
            classes="elsevier-table", border=0, index=False, justify="center"
        )

        css_style = f"""
        <style>
            body {{ font-family: "{FONT_SERIF}"; padding: 30px; max-width: 1200px; margin: auto; background-color: #fcfcfc; }}
            .report-container {{ background-color: white; padding: 30px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }}
            h2, h3, h4 {{ text-align: center; color: #333; }}
            .elsevier-table {{ width: 100%; border-collapse: collapse; margin-top: 20px; font-size: 11pt; }}
            .elsevier-table thead tr {{ border-top: 2px solid black; border-bottom: 1px solid black; }}
            .elsevier-table th, .elsevier-table td {{ padding: 8px; text-align: right; }}
            .elsevier-table td:first-child, .elsevier-table th:first-child {{ text-align: left; font-weight: bold; width: 220px; }}
            .elsevier-table tbody tr:last-child td {{ border-bottom: 2px solid black; }}
            .elsevier-table tbody tr:hover {{ background-color: #f5f5f5; }}
        </style>
        """

        report_html = f"""
        <!DOCTYPE html>
        <html><head><meta charset="utf-8">{css_style}</head><body>
            <div class="report-container">
                <h2>{title}</h2>
                {html_plots}
                <br><br>
                <h3>Table 1: Out-of-Sample Performance Metrics across Variations</h3>
                {main_table_html}
            </div>
        </body></html>
        """

        with open(
            paper_out_dir / f"{prefix}_interactive_report.html", "w", encoding="utf-8"
        ) as f:
            f.write(report_html)
        logger.info(f"Generated HTML report: {prefix}_interactive_report.html")

    save_combined_report(
        sensitivity_results, sensitivity_plots, "sensitivity", "Sensitivity Analysis"
    )
    save_combined_report(
        assumptions_results, mechanism_plots, "mechanism", "Assumptions Verification"
    )


if __name__ == "__main__":
    generate_reports(FOLDER, BASELINE)
