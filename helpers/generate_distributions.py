"""Script to generate grid-search distributions, including PDF plots and summary parquet."""

import os
import sys
import math
import time

import pandas as pd
import yaml
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
from pathlib import Path

from modules.utils.logger import get_logger

logger = get_logger(__name__)

FOLDER = "Baseline Optimization"

TARGET_METRIC = "Sortino Ratio"

PARAMS_TO_PLOT = [
    "Entry Threshold",
    "Exit Threshold",
    "Stop Loss",
    "Z-Score Window",
    "Pairs",
]

ELSEVIER_FONT = "Arial, sans-serif"
FONT_SIZE_TICK = 10
FONT_SIZE_LABEL = 12
FONT_SIZE_TITLE = 13
COLOR_BLACK = "black"

PDF_WIDTH = 720
PDF_HEIGHT = 300

current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
sys.path.append(str(project_root))


def plot_distributions(
    df: pd.DataFrame,
    out_dir: Path,
    target_metric: str,
    run_name: str,
):
    pio.defaults.default_format = "pdf"
    pio.defaults.mathjax = None

    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Generating individual PDFs for '{run_name}' in: {out_dir}")

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
        tickformat=".2f",
        tickfont=dict(family=ELSEVIER_FONT, size=FONT_SIZE_TICK, color=COLOR_BLACK),
        title_font=dict(family=ELSEVIER_FONT, size=FONT_SIZE_LABEL, color=COLOR_BLACK),
        showgrid=True,
        gridcolor="#E5E5E5",
        gridwidth=0.5,
        zeroline=True,
        zerolinecolor="#E5E5E5",
        zerolinewidth=0.5,
    )

    legend_style = dict(
        yanchor="bottom",
        y=0.05,
        xanchor="right",
        x=0.99,
        font=dict(family=ELSEVIER_FONT, size=FONT_SIZE_TICK, color=COLOR_BLACK),
        bgcolor="rgba(255, 255, 255, 0.8)",
        bordercolor=COLOR_BLACK,
        borderwidth=0.2,
    )

    valid_params = [
        p for p in PARAMS_TO_PLOT if p in df.columns and df[p].nunique(dropna=False) > 1
    ]

    for param in valid_params:
        df_valid = df.copy()
        df_valid[target_metric] = pd.to_numeric(
            df_valid[target_metric], errors="coerce"
        )
        df_valid = df_valid.dropna(subset=[target_metric])

        if df_valid[param].nunique(dropna=False) <= 1:
            logger.warning(
                f"Not enough variation to plot {param} for {run_name}. Skipping."
            )
            continue

        df_valid[f"{param}_str"] = (
            df_valid[param]
            .astype(str)
            .replace({"nan": "None", "NaN": "None", "<NA>": "None"})
        )

        fig_pdf = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=[
                f"Panel A: Distribution by {param}",
                f"Panel B: Median & Mean by {param}",
            ],
            horizontal_spacing=0.08,
        )

        counts_full = df_valid[f"{param}_str"].value_counts()

        def numeric_sort_key(v):
            if str(v).lower() == "none":
                return 1, float("inf")
            try:
                return 0, float(v)
            except ValueError:
                return 0, str(v)

        unique_vals = sorted(counts_full.index, key=numeric_sort_key)
        max_obs = counts_full.max() if not counts_full.empty else 1

        ordered_x_labels = []
        medians = []
        means = []

        for val in unique_vals:
            subset = df_valid[df_valid[f"{param}_str"] == val]
            n_obs = len(subset)
            box_width = 0.8 * math.sqrt(n_obs / max_obs) if max_obs > 0 else 0.8

            try:
                formatted_val = f"{float(val):.2f}"
            except ValueError:
                formatted_val = str(val)

            x_label = formatted_val
            ordered_x_labels.append(x_label)

            medians.append(subset[target_metric].median())
            means.append(subset[target_metric].mean())

            trace_box = go.Box(
                y=subset[target_metric],
                x=[x_label] * len(subset),
                name=x_label,
                width=box_width,
                offsetgroup="1",
                fillcolor="#E0E0E0",
                line=dict(color=COLOR_BLACK, width=1.0),
                marker=dict(color=COLOR_BLACK, size=4, symbol="circle-open"),
                boxpoints="outliers",
                showlegend=False,
            )
            fig_pdf.add_trace(trace_box, row=1, col=1)

        trace_median = go.Scatter(
            x=ordered_x_labels,
            y=medians,
            mode="lines",
            name="Median",
            line=dict(color=COLOR_BLACK, width=1.0),
            showlegend=True,
        )
        trace_mean = go.Scatter(
            x=ordered_x_labels,
            y=means,
            mode="lines",
            name="Mean",
            line=dict(color=COLOR_BLACK, width=1.0, dash="dash"),
            showlegend=True,
        )

        fig_pdf.add_trace(trace_median, row=1, col=2)
        fig_pdf.add_trace(trace_mean, row=1, col=2)

        for c in [1, 2]:
            fig_pdf.update_xaxes(
                **axis_style_x,
                categoryorder="array",
                categoryarray=ordered_x_labels,
                title_text=param,
                row=1,
                col=c,
                range=[-0.5, len(ordered_x_labels) - 0.5],
            )
            fig_pdf.update_yaxes(**axis_style_y, title_text=target_metric, row=1, col=c)

        fig_pdf.update_layout(
            width=PDF_WIDTH,
            height=PDF_HEIGHT,
            font=dict(family=ELSEVIER_FONT, size=FONT_SIZE_LABEL, color=COLOR_BLACK),
            plot_bgcolor="white",
            paper_bgcolor="white",
            showlegend=True,
            legend=legend_style,
            margin=dict(t=45, b=40, l=45, r=10),
        )

        for annotation in fig_pdf["layout"]["annotations"]:
            annotation["font"] = dict(
                family=ELSEVIER_FONT, size=FONT_SIZE_TITLE, color=COLOR_BLACK
            )
            annotation["yshift"] = 5

        pdf_path = out_dir / f"{param} {run_name}.pdf"
        try:
            fig_pdf.write_image(str(pdf_path), format="pdf")
            logger.info(f"PDF saved: {pdf_path.name}")
        except Exception as e:
            logger.warning(
                f"Initial save failed for {pdf_path.name}. Killing kaleido and retrying... {e}"
            )
            os.system("taskkill /F /IM kaleido.exe /T >nul 2>&1")
            time.sleep(1)

            max_retries = 3
            for attempt in range(max_retries):
                try:
                    fig_pdf.write_image(str(pdf_path), format="pdf")
                    logger.info(f"PDF saved on retry {attempt + 1}: {pdf_path.name}")
                    break
                except Exception as retry_e:
                    if attempt < max_retries - 1:
                        logger.warning(
                            f"Retry {attempt + 1} failed for {pdf_path.name}. Killing kaleido again..."
                        )
                        os.system("taskkill /F /IM kaleido.exe /T >nul 2>&1")
                        time.sleep(2)
                    else:
                        logger.error(
                            f"Failed to save {pdf_path.name} after {max_retries} retries: {retry_e}"
                        )


def generate_distributions(run_dir: Path, output_dir: Path):
    run_name = run_dir.name

    logger.info(f"Extracting metrics and building DataFrames for {run_name}...")
    all_results_list = []

    config_path = run_dir / ".hydra" / "config.yaml"
    stats_files = list(run_dir.glob("*/*/stats_multi_pair_*.parquet"))

    if not stats_files:
        logger.warning(f"Global stats not found in {run_dir}. Skipping {run_name}.")
        return

    for stats_file in stats_files:
        try:
            cfg_local = stats_file.parent / ".hydra" / "config.yaml"
            cfg_parent = stats_file.parents[1] / ".hydra" / "config.yaml"

            if cfg_local.exists():
                cfg_to_load = cfg_local
            elif cfg_parent.exists():
                cfg_to_load = cfg_parent
            elif config_path.exists():
                cfg_to_load = config_path
            else:
                logger.error(f"Config not found for {stats_file}")
                continue

            with open(cfg_to_load, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)

            perf = config.get("performance", {})
            ps = config.get("pair_selection", {})

            entry = perf.get("entry_threshold", None)
            exit_t = perf.get("exit_threshold", None)
            stop_loss = perf.get("stop_loss", None)
            z_score_window = perf.get("z_score_window", None)
            top_n = ps.get("top_n", None)

            stats_df = pd.read_parquet(stats_file)

            def get_metric(metric_name, col_type):
                row = stats_df[stats_df["metric"] == metric_name]
                if not row.empty:
                    return row[col_type].iloc[0]
                return None

            row_data = {
                "Run_ID": (
                    stats_file.parent.name
                    if stats_file.parent.name.isdigit()
                    else run_dir.name
                ),
                "Entry Threshold": entry,
                "Exit Threshold": exit_t,
                "Stop Loss": stop_loss,
                "Z-Score Window": z_score_window,
                "Pairs": top_n,
                "CAGR": get_metric("cagr", "net"),
                "Annual Volatility": get_metric("volatility_annual", "net"),
                "Max Drawdown": get_metric("max_drawdown", "net"),
                "Total Trades": int(get_metric("win_count", "net") or 0)
                + int(get_metric("lose_count", "net") or 0),
                "Avg Trade Duration": get_metric("avg_trade_duration", "net"),
                "Avg Trade Return": get_metric("avg_trade_return", "net"),
                "Sharpe Ratio": get_metric("sharpe_ratio_annual", "net"),
                "Sortino Ratio": get_metric("sortino_ratio_annual", "net"),
                "Calmar Ratio": get_metric("calmar_ratio", "net"),
            }

            all_results_list.append(row_data)

        except Exception as e:
            logger.error(f"Error processing stats for {stats_file}: {e}")

    logger.info(f"Saving results and generating plots for {run_name}...")
    df_summary = pd.DataFrame(all_results_list)

    df_summary = df_summary.sort_values(by=TARGET_METRIC, ascending=False).reset_index(
        drop=True
    )

    cols_to_str = [
        "Entry Threshold",
        "Exit Threshold",
        "Stop Loss",
        "Z-Score Window",
        "Pairs",
    ]
    for col in cols_to_str:
        if col in df_summary.columns:
            df_summary[col] = df_summary[col].astype(str)

    output_parquet = output_dir / f"{run_name}.parquet"
    df_summary.to_parquet(output_parquet, engine="pyarrow", index=False)
    logger.info(f"Saved global summary to: {output_parquet}")

    plot_distributions(
        df=df_summary,
        out_dir=output_dir,
        target_metric=TARGET_METRIC,
        run_name=run_name,
    )


if __name__ == "__main__":
    experiment_dir = project_root / "results" / FOLDER
    output_dir = experiment_dir / "distributions"

    if not experiment_dir.exists():
        logger.error(f"Directory '{experiment_dir}' does not exist. Cannot proceed.")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    found_subdirs = [
        d for d in experiment_dir.iterdir() if d.is_dir() and d.name != "distributions"
    ]

    if not found_subdirs:
        logger.warning(f"No subdirectories found in {experiment_dir}.")
    else:
        for subdir in sorted(found_subdirs):
            generate_distributions(run_dir=subdir, output_dir=output_dir)

    logger.info("Ending...")
    os._exit(0)
