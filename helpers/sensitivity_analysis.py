import yaml
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
from plotly.subplots import make_subplots

from modules.utils.logger import get_logger

logger = get_logger(__name__)

FOLDER = "assumptions_verification"

IS_BASE_DIR = "is"
OOS_BASE_DIR = "oos"

BASELINE = {
    "IS": "winner_is",
    "OOS": "winner_oos",
}

LEVERAGE = 10.0

FONT_SANS = "Arial, sans-serif"
FONT_SERIF = "Times New Roman, serif"

SENSITIVITY_PARAMS = [
    "entry_threshold",
    "exit_threshold",
    "stop_loss",
    "z_score_window",
    "top_n",
]

ASSUMPTIONS = [
    "beta_hedge",
    "freeze_std",
    "delayed_entry",
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
    "TDA-Sortino",
]

NAME_MAP = {
    "entry_threshold": "Entry Threshold",
    "exit_threshold": "Exit Threshold",
    "stop_loss": "Stop Loss",
    "z_score_window": "Z-Score Window",
    "top_n": "Pairs",
    "beta_hedge": "Beta Hedge",
    "freeze_std": "Freeze Std",
    "delayed_entry": "Delayed Entry",
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
    "TDA-Sortino": "{:.4f}",
}


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
        stats_files = list(run_dir.glob("stats_multi_pair_*.parquet"))
        ts_files = list(run_dir.glob("returns_multi_pair_*.parquet"))

        if not (config_path.exists() and stats_files and ts_files):
            return None

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)

            stats_df = pd.read_parquet(stats_files[0])
            ts_df = pd.read_parquet(ts_files[0])

            def get_cfg(key):
                return config.get(
                    key,
                    config.get("performance", {}).get(
                        key, config.get("pair_selection", {}).get(key)
                    ),
                )

            params = {k: get_cfg(k) for k in (SENSITIVITY_PARAMS + ASSUMPTIONS)}

            def get_stat(metric):
                row = stats_df[stats_df["metric"] == metric]
                return row["net"].iloc[0] if not row.empty else None

            metrics = {
                "CAGR": get_stat("cagr"),
                "Annual Volatility": get_stat("volatility_annual"),
                "Max Drawdown": get_stat("max_drawdown"),
                "Win Count": get_stat("win_count"),
                "Lose Count": get_stat("lose_count"),
                "Win Rate": get_stat("win_rate"),
                "Avg Win": get_stat("avg_win_return"),
                "Avg Lose": get_stat("avg_lose_return"),
                "Avg Trade Return": get_stat("avg_trade_return"),
                "Avg Trade Duration": get_stat("avg_trade_duration"),
                "Sharpe Ratio": get_stat("sharpe_ratio_annual"),
                "Sortino Ratio": get_stat("sortino_ratio_annual"),
                "Calmar Ratio": get_stat("calmar_ratio_annual"),
                "TDA-Sortino": get_stat("tda_sortino"),
            }

            ret_series = (
                (ts_df["total_pnl"] - ts_df["total_fees"]) * LEVERAGE / initial_cash
            )

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
        "Variation": "Baseline Model",
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

    for oos_run_dir in oos_base_dir.iterdir():
        if not oos_run_dir.is_dir() or oos_run_dir.name == baseline_dict["OOS"]:
            continue

        var_name = oos_run_dir.name
        is_run_dir = is_base_dir / var_name

        if not is_run_dir.exists():
            logger.warning(
                f"Variation {var_name} found in OOS, but missing in IS. Skipping."
            )
            continue

        run_is = get_run_data(is_run_dir)
        run_oos = get_run_data(oos_run_dir)

        if not run_is or not run_oos:
            continue

        diffs = [
            k
            for k in (SENSITIVITY_PARAMS + ASSUMPTIONS)
            if is_different(run_oos["params"][k], base_oos["params"][k])
        ]

        if len(diffs) == 1:
            p_name = diffs[0]
            val = run_oos["params"][p_name]
            var_label = f"{NAME_MAP.get(p_name, p_name)} = {val}"

            is_series = run_is["ret_series"] - run_is["ret_series"].iloc[0]
            oos_series = run_oos["ret_series"] - run_oos["ret_series"].iloc[0]

            plot_item = {
                "series_is": is_series,
                "series_oos": oos_series,
                "name": var_label,
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

    def save_combined_report(results, plots, prefix, title, variant_color):
        if len(results) <= 1:
            logger.info(f"Skipping '{title}' (only baseline found).")
            return

        df_raw = pd.DataFrame(results).sort_values(by=["Group", "Variation"])
        df_raw.to_parquet(paper_out_dir / f"{prefix}_stats.parquet", index=False)

        fig = make_subplots(
            rows=1,
            cols=2,
            shared_yaxes=True,
            horizontal_spacing=0.03,
            subplot_titles=[
                "Panel A: In-Sample Performance",
                "Panel B: Out-of-Sample Performance",
            ],
        )

        for p in plots:
            fig.add_trace(
                go.Scatter(
                    x=p["series_is"].index,
                    y=p["series_is"].values,
                    mode="lines",
                    line=dict(color=variant_color, width=1.5),
                    opacity=0.3,
                    name=p["name"],
                    legendgroup=p["name"],
                    hovertemplate=f"<b>{p['name']} (IS)</b><br>Return: %{{y:.2%}}<extra></extra>",
                ),
                row=1,
                col=1,
            )

            fig.add_trace(
                go.Scatter(
                    x=p["series_oos"].index,
                    y=p["series_oos"].values,
                    mode="lines",
                    line=dict(color=variant_color, width=1.5),
                    opacity=0.3,
                    name=p["name"],
                    legendgroup=p["name"],
                    showlegend=False,
                    hovertemplate=f"<b>{p['name']} (OOS)</b><br>Return: %{{y:.2%}}<extra></extra>",
                ),
                row=1,
                col=2,
            )

        fig.add_trace(
            go.Scatter(
                x=base_is_series.index,
                y=base_is_series.values,
                mode="lines",
                line=dict(color="black", width=2.5),
                name="Baseline Model",
                legendgroup="Baseline",
                hovertemplate="<b>Baseline (IS)</b><br>Return: %{y:.2%}<extra></extra>",
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=base_oos_series.index,
                y=base_oos_series.values,
                mode="lines",
                line=dict(color="black", width=2.5),
                name="Baseline Model",
                legendgroup="Baseline",
                showlegend=False,
                hovertemplate="<b>Baseline (OOS)</b><br>Return: %{y:.2%}<extra></extra>",
            ),
            row=1,
            col=2,
        )

        fig.update_layout(
            template="plotly_white",
            font=dict(family=FONT_SANS, color="black"),
            margin=dict(t=50, b=50, l=70, r=20),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.08,
                xanchor="center",
                x=0.5,
                font=dict(size=11),
            ),
            width=1000,
            height=450,
        )

        fig.update_yaxes(
            title_text="Cumulative Return",
            tickformat=".0%",
            showline=True,
            linewidth=1,
            linecolor="black",
            gridcolor="#e5e5e5",
            mirror=True,
            zeroline=True,
            zerolinecolor="black",
            row=1,
            col=1,
        )
        fig.update_yaxes(
            tickformat=".0%",
            showline=True,
            linewidth=1,
            linecolor="black",
            gridcolor="#e5e5e5",
            mirror=True,
            zeroline=True,
            zerolinecolor="black",
            row=1,
            col=2,
        )
        fig.update_xaxes(
            showline=True,
            linewidth=1,
            linecolor="black",
            gridcolor="#e5e5e5",
            mirror=True,
            row=1,
            col=1,
        )
        fig.update_xaxes(
            showline=True,
            linewidth=1,
            linecolor="black",
            gridcolor="#e5e5e5",
            mirror=True,
            row=1,
            col=2,
        )

        try:
            pdf_path = pdf_dir / f"{prefix}_spaghetti_subplots.pdf"
            fig.write_image(str(pdf_path), format="pdf", engine="kaleido")
            logger.info(f"PDF saved: {pdf_path}")
        except Exception as e:
            logger.error(f"Error during PDF export: {e}")

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
            h2, h3 {{ text-align: center; color: #333; }}
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
                <div>{fig.to_html(full_html=False, include_plotlyjs='cdn')}</div>
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
        sensitivity_results,
        sensitivity_plots,
        "sensitivity",
        "Sensitivity Analysis",
        "#1f77b4",
    )
    save_combined_report(
        assumptions_results,
        mechanism_plots,
        "mechanism",
        "Assumptions Verification",
        "#d62728",
    )


if __name__ == "__main__":
    generate_reports(FOLDER, BASELINE)
