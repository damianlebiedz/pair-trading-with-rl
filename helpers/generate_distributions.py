import yaml
import math
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

from modules.utils.logger import get_logger

logger = get_logger(__name__)

FOLDER = "stage 1 zoom"
TOP_N_ROWS = 10
TARGET_METRIC = "Sortino Ratio"

FILTER = {
    # "Entry Threshold": 3.0,
    "Exit Threshold": 0.0,
    "Stop Loss": 2.0,
    "Z-Score Window": 168,
    "Pairs": 20,
}

STAR = {
    "Entry Threshold": 3.0,
    # "Stop Loss": 2.0,
}

HIGHLIGHT_RANGE = {
    # "Entry Threshold": [2.75, 3.25],
    # "Stop Loss": [1.75, 2.25]
}

PARAMS_TO_PLOT = [
    "Entry Threshold",
    "Exit Threshold",
    "Stop Loss",
    "Z-Score Window",
    "Pairs",
]

SELECTED_METRICS = [
    "Entry Threshold",
    "Exit Threshold",
    "Stop Loss",
    "Z-Score Window",
    "Pairs",
    "Total Trades",
    "Avg Trade Duration",
    "Max Drawdown",
    "Sharpe Ratio",
    "Calmar Ratio",
    "TDA-Sortino",
]

RENAME_MAP = {
    "Entry Threshold": "Entry Threshold",
    "Exit Threshold": "Exit Threshold",
    "Stop Loss": "Stop Loss",
    "Z-Score Window": "Z-Score Window",
    "Pairs": "Pairs",
    "Total Trades": "Total Trades",
    "Avg Trade Duration": "Avg Trade Duration",
    "Max Drawdown": "Max Drawdown",
    "Sharpe Ratio": "Sharpe Ratio",
    "Sortino Ratio Median": "Sortino Ratio Median",
    "Calmar Ratio": "Calmar Ratio",
    "TDA-Sortino": "TDA-Sortino",
}

FORMAT_MAP = {}

ELSEVIER_FONT = "Arial, sans-serif"
FONT_SIZE_TICK = 10
FONT_SIZE_LABEL = 12
FONT_SIZE_TITLE = 13
COLOR_BLACK = "black"

PDF_WIDTH = 720
PDF_HEIGHT = 300


def plot_distribution(
    df: pd.DataFrame,
    title: str,
    out_file: Path,
    target_metric: str,
    filters: dict = None,
    stars: dict = None,
    highlight_ranges: dict = None,
):
    valid_params = [
        p for p in PARAMS_TO_PLOT if p in df.columns and df[p].nunique(dropna=False) > 1
    ]

    if not valid_params:
        logger.error(f" No variable parameters to plot for: {title}.")
        return

    pdf_dir = out_file.with_suffix("")
    pdf_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Created directory for individual PDFs: {pdf_dir}")

    n_params = len(valid_params)
    rows = n_params
    cols = 2

    filter_annotation = ""
    if filters:
        f_str = ", ".join([f"{k} = {v}" for k, v in filters.items()])
        filter_annotation = f"<br><span style='font-size: 9px; font-weight: normal; color: #444444;'>({f_str})</span>"

    subplot_titles = []
    for p in valid_params:
        subplot_titles.append(f"Panel A: Distribution by {p}{filter_annotation}")
        subplot_titles.append(f"Panel B: Median & Mean by {p}{filter_annotation}")

    fig_html = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=subplot_titles,
        vertical_spacing=0.15,
        horizontal_spacing=0.08,
    )

    df = df.copy()
    df[target_metric] = pd.to_numeric(df[target_metric], errors="coerce")

    for param in valid_params:
        df[f"{param}_str"] = (
            df[param]
            .astype(str)
            .replace({"nan": "None", "NaN": "None", "<NA>": "None"})
        )

    df_valid = df.dropna(subset=[target_metric]).copy()

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
        yanchor="top",
        y=0.98,
        xanchor="right",
        x=0.99,
        font=dict(family=ELSEVIER_FONT, size=FONT_SIZE_TICK, color=COLOR_BLACK),
        bgcolor="rgba(255, 255, 255, 0.8)",
        bordercolor=COLOR_BLACK,
        borderwidth=0.5,
    )

    optimum_html_legend_added = False

    for i, param in enumerate(valid_params):
        r = i + 1
        counts_full = df[f"{param}_str"].value_counts()

        def numeric_sort_key(v):
            if str(v).lower() == "none":
                return 1, float("inf")
            try:
                return 0, float(v)
            except ValueError:
                return 0, str(v)

        unique_vals = sorted(counts_full.index, key=numeric_sort_key)
        counts_valid = df_valid[f"{param}_str"].value_counts()
        max_obs = counts_valid.max() if not counts_valid.empty else 1

        ordered_x_labels = []
        traces_box = []
        medians = []
        means = []
        star_x_index = None

        for val in unique_vals:
            subset = df_valid[df_valid[f"{param}_str"] == val]
            n_obs = len(subset)
            box_width = 0.8 * math.sqrt(n_obs / max_obs) if max_obs > 0 else 0.8

            try:
                formatted_val = f"{float(val):.2f}"
            except ValueError:
                formatted_val = str(val)

            display_val = formatted_val
            is_starred = False
            if stars and param in stars and str(stars[param]) == val:
                display_val = f"{formatted_val}*"
                is_starred = True

            x_label = f"{display_val}"
            ordered_x_labels.append(x_label)

            if is_starred:
                star_x_index = len(ordered_x_labels) - 1

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
            traces_box.append(trace_box)
            fig_html.add_trace(trace_box, row=r, col=1)

        trace_median = go.Scatter(
            x=ordered_x_labels,
            y=medians,
            mode="lines",
            name="Median",
            line=dict(color=COLOR_BLACK, width=1.0),
            showlegend=True if i == 0 else False,
        )

        trace_mean = go.Scatter(
            x=ordered_x_labels,
            y=means,
            mode="lines",
            name="Mean",
            line=dict(color=COLOR_BLACK, width=1.0, dash="dash"),
            showlegend=True if i == 0 else False,
        )

        fig_html.add_trace(trace_median, row=r, col=2)
        fig_html.add_trace(trace_mean, row=r, col=2)

        if star_x_index is not None:
            show_opt_html = not optimum_html_legend_added

            fig_html.add_vline(
                x=star_x_index,
                line_width=1.0,
                line_dash="dot",
                line_color=COLOR_BLACK,
                row=r,
                col=2,
                name="Optimum*",
                showlegend=show_opt_html,
            )
            optimum_html_legend_added = True

        for c in [1, 2]:
            fig_html.update_xaxes(
                **axis_style_x,
                categoryorder="array",
                categoryarray=ordered_x_labels,
                title_text=param,
                row=r,
                col=c,
                range=[-0.5, len(ordered_x_labels) - 0.5],
            )
            fig_html.update_yaxes(
                **axis_style_y, title_text=target_metric, row=r, col=c
            )

        fig_pdf = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=[
                f"Panel A: Distribution by {param}{filter_annotation}",
                f"Panel B: Median & Mean by {param}{filter_annotation}",
            ],
            horizontal_spacing=0.08,
        )

        for tb in traces_box:
            fig_pdf.add_trace(tb, row=1, col=1)

        trace_median_pdf = go.Scatter(
            x=ordered_x_labels,
            y=medians,
            mode="lines",
            name="Median",
            line=dict(color=COLOR_BLACK, width=1.0),
            showlegend=True,
        )
        trace_mean_pdf = go.Scatter(
            x=ordered_x_labels,
            y=means,
            mode="lines",
            name="Mean",
            line=dict(color=COLOR_BLACK, width=1.0, dash="dash"),
            showlegend=True,
        )

        fig_pdf.add_trace(trace_median_pdf, row=1, col=2)
        fig_pdf.add_trace(trace_mean_pdf, row=1, col=2)

        if star_x_index is not None:
            fig_pdf.add_vline(
                x=star_x_index,
                line_width=1.0,
                line_dash="dot",
                line_color=COLOR_BLACK,
                row=1,
                col=2,
                name="Optimum*",
                showlegend=True,
            )

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

        if highlight_ranges and param in highlight_ranges:
            min_val, max_val = highlight_ranges[param]
            range_indices = [
                idx
                for idx, val_str in enumerate(unique_vals)
                if (
                    v := (
                        float(val_str)
                        if val_str.replace(".", "", 1).isdigit()
                        else None
                    )
                )
                is not None
                and min_val <= v <= max_val
            ]

            if range_indices:
                start_idx, end_idx = min(range_indices), max(range_indices)
                for c in [1, 2]:
                    fig_html.add_vrect(
                        x0=start_idx,
                        x1=end_idx,
                        fillcolor="rgba(200, 200, 200, 0.3)",
                        layer="below",
                        line_width=0,
                        row=r,
                        col=c,
                    )
                    fig_pdf.add_vrect(
                        x0=start_idx,
                        x1=end_idx,
                        fillcolor="rgba(200, 200, 200, 0.3)",
                        layer="below",
                        line_width=0,
                        row=1,
                        col=c,
                    )

        fig_pdf.update_layout(
            width=PDF_WIDTH,
            height=PDF_HEIGHT,
            font=dict(family=ELSEVIER_FONT, size=FONT_SIZE_LABEL, color=COLOR_BLACK),
            plot_bgcolor="white",
            paper_bgcolor="white",
            showlegend=True,
            legend=legend_style,
            margin=dict(t=40, b=50, l=0, r=5),
        )

        for annotation in fig_pdf["layout"]["annotations"]:
            annotation["font"] = dict(
                family=ELSEVIER_FONT, size=FONT_SIZE_TITLE, color=COLOR_BLACK
            )

        pdf_path = pdf_dir / f"{param}.pdf"
        try:
            fig_pdf.write_image(str(pdf_path), format="pdf")
            logger.info(f"Successfully saved Elsevier-styled PDF: {pdf_path.name}")
        except Exception as e:
            logger.error(f"Failed to save PDF for {param}. Error: {e}")
            logger.error("Try running: pip install -U kaleido")

    fig_html.update_layout(
        height=PDF_HEIGHT * rows if rows > 0 else 400,
        width=PDF_WIDTH,
        title_text="",
        font=dict(family=ELSEVIER_FONT, size=FONT_SIZE_LABEL, color=COLOR_BLACK),
        plot_bgcolor="white",
        paper_bgcolor="white",
        showlegend=True,
        legend=legend_style,
        margin=dict(t=40, b=40, l=45, r=10),
    )

    for annotation in fig_html["layout"]["annotations"]:
        annotation["font"] = dict(
            family=ELSEVIER_FONT, size=FONT_SIZE_TITLE, color=COLOR_BLACK
        )

    fig_html.write_html(str(out_file))
    logger.info(f"Saved combined HTML dashboard: {out_file}")
    logger.info(f"Saved individual PDFs in: {pdf_dir}/")


def generate_distributions():
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    run_dir = project_root / "results" / f"{FOLDER}"
    output_dir = project_root / "results" / f"{FOLDER} distribution"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not run_dir.exists():
        raise ValueError(f"Directory {run_dir} not found. Run analysis first.")

    logger.info("Extracting metrics and building DataFrames...")
    all_results_list = []

    config_path = run_dir / ".hydra" / "config.yaml"
    stats_files = list(run_dir.glob("*/*/stats_multi_pair_*.parquet"))

    if not stats_files:
        raise ValueError(f"Global stats not found in {run_dir}")

    if not config_path.exists() and not list(run_dir.rglob("config.yaml")):
        logger.warning(f"Config files not found in {run_dir}")

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
                "Sortino Ratio Median": get_metric("sortino_annual_median", "net"),
                "Calmar Ratio": get_metric("calmar_ratio", "net"),
                "TDA-Sortino": get_metric("tda_sortino", "net"),
                "Custom Score": get_metric("sortino_ratio_annual", "net") ** 2,
            }

            all_results_list.append(row_data)

        except Exception as e:
            logger.error(f"Error processing stats for {stats_file}: {e}")

    logger.info("Saving results and generating plots...")
    df_summary = pd.DataFrame(all_results_list)

    for col, value in FILTER.items():
        df_summary = df_summary[df_summary[col] == value]

    df_summary = df_summary.sort_values(by=TARGET_METRIC, ascending=False).reset_index(
        drop=True
    )

    available_metrics = [m for m in SELECTED_METRICS if m in df_summary.columns]
    df_tex = df_summary[available_metrics].copy()

    for col in df_tex.columns:
        if col in FORMAT_MAP:
            fmt = FORMAT_MAP[col]
            df_tex[col] = df_tex[col].apply(
                lambda x: fmt.format(x) if pd.notna(x) else "-"
            )
        else:
            if pd.api.types.is_numeric_dtype(df_tex[col]):
                df_tex[col] = df_tex[col].apply(
                    lambda x: f"{x:.4f}" if pd.notna(x) else "-"
                )

    df_tex = df_tex.rename(columns=RENAME_MAP)
    tex_path = output_dir / "top_grid_search_results.tex"

    with open(tex_path, "w") as f:
        f.write(
            df_tex.head(TOP_N_ROWS).to_latex(
                index=False, column_format="c" * len(df_tex.columns), escape=False
            )
        )
    logger.info(f"Saved LaTeX table (Top {TOP_N_ROWS}): {tex_path}")

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

    output_parquet = output_dir / "summary_distribution.parquet"
    df_summary.to_parquet(output_parquet, engine="pyarrow", index=False)
    logger.info(f"Saved summary: {output_parquet}")

    output_html = output_dir / "plots_distribution.html"

    plot_distribution(
        df_summary,
        title="TDA-Sortino Distribution",
        out_file=output_html,
        target_metric=TARGET_METRIC,
        filters=FILTER,
        stars=STAR,
        highlight_ranges=HIGHLIGHT_RANGE,
    )


if __name__ == "__main__":
    generate_distributions()
