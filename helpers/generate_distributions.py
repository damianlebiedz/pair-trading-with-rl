import yaml
import math
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

from modules.utils.logger import get_logger

logger = get_logger(__name__)

FOLDER = "grid"
TOP_N_ROWS = 10

SELECTED_METRICS = [
    "Entry",
    "Exit",
    "SL",
    "Z-Score Window",
    "Pairs",
    "Total Trades",
    "Avg Trade Duration",
    "Max Drawdown",
    "Sortino Ratio Median",
    "TDA-Sortino",
]

RENAME_MAP = {
    "Entry": "Entry",
    "Exit": "Exit",
    "SL": "Stop Loss",
    "Z-Score Window": "Z-Score Window",
    "Pairs": "Pairs",
    "Total Trades": "Total Trades",
    "Avg Trade Duration": "Avg Trade Duration",
    "Max Drawdown": "Max Drawdown",
    "Sortino Ratio Median": "Sortino Ratio Median",
    "TDA-Sortino": "TDA-Sortino",
}

FORMAT_MAP = {}


def plot_distribution(
    df: pd.DataFrame,
    title: str,
    out_file: Path,
    params_to_plot: list = None,
    target_metric: str = "TDA-Sortino",
):
    if params_to_plot is None:
        params_to_plot = [
            "Entry",
            "Exit",
            "SL",
            "Z-Score Window",
            "Beta Hedge",
            "Pairs",
        ]

    valid_params = [
        p for p in params_to_plot if p in df.columns and df[p].nunique(dropna=False) > 1
    ]

    if not valid_params:
        logger.error(f" No variable parameters to plot for: {title}.")
        return

    pdf_dir = out_file.with_suffix("")
    pdf_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Created directory for individual PDFs: {pdf_dir}")

    n_params = len(valid_params)
    cols = 2
    rows = math.ceil(n_params / cols)

    fig_html = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=[
            f"Panel {chr(65 + i)}: Distribution by {p}"
            for i, p in enumerate(valid_params)
        ],
        vertical_spacing=0.15,
        horizontal_spacing=0.1,
    )

    df = df.copy()
    df[target_metric] = pd.to_numeric(df[target_metric], errors="coerce")

    font_family = "Arial, sans-serif"
    pdf_width = 380
    pdf_height = 320

    for i, param in enumerate(valid_params):
        r = (i // cols) + 1
        c = (i % cols) + 1

        df[f"{param}_str"] = (
            df[param]
            .astype(str)
            .replace({"nan": "None", "NaN": "None", "<NA>": "None"})
        )

        counts = df[f"{param}_str"].value_counts()
        unique_vals = sorted(counts.index, key=lambda x: (x.lower() == "none", x))
        max_count = counts.max() if not counts.empty else 1

        ordered_x_labels = []
        traces = []

        for val in unique_vals:
            subset = df[df[f"{param}_str"] == val]
            n_obs = counts[val]

            box_width = 0.8 * math.sqrt(n_obs / max_count) if max_count > 0 else 0.8
            x_label = f"{val}<br>(N={n_obs})"
            ordered_x_labels.append(x_label)

            trace = go.Box(
                y=subset[target_metric],
                x=[x_label] * len(subset),
                name=x_label,
                width=box_width,
                fillcolor="#E0E0E0",
                line=dict(color="black", width=1.5),
                marker=dict(color="black", size=5, symbol="circle-open"),
                boxpoints="outliers",
                showlegend=False,
            )
            traces.append(trace)
            fig_html.add_trace(trace, row=r, col=c)

        fig_html.add_hline(
            y=0,
            line_dash="dash",
            line_color="black",
            line_width=1,
            opacity=0.7,
            row=r,
            col=c,
        )

        fig_html.update_xaxes(
            categoryorder="array",
            categoryarray=ordered_x_labels,
            title_text=param,
            row=r,
            col=c,
            showline=True,
            linewidth=1,
            linecolor="black",
            mirror=True,
            ticks="inside",
            tickcolor="black",
            tickwidth=1,
            showgrid=False,
        )
        fig_html.update_yaxes(
            title_text=target_metric,
            row=r,
            col=c,
            showline=True,
            linewidth=1,
            linecolor="black",
            mirror=True,
            ticks="inside",
            tickcolor="black",
            tickwidth=1,
            showgrid=True,
            gridcolor="#E5E5E5",
            gridwidth=1,
        )

        fig_pdf = go.Figure(data=traces)
        fig_pdf.add_hline(
            y=0, line_dash="dash", line_color="black", line_width=1, opacity=0.7
        )

        fig_pdf.update_xaxes(
            categoryorder="array",
            categoryarray=ordered_x_labels,
            title_text=param,
            showline=True,
            linewidth=1,
            linecolor="black",
            mirror=True,
            ticks="inside",
            tickcolor="black",
            tickwidth=1,
            showgrid=False,
            title_font=dict(size=12, color="black"),
            tickfont=dict(size=10, color="black"),
        )
        fig_pdf.update_yaxes(
            title_text=target_metric,
            showline=True,
            linewidth=1,
            linecolor="black",
            mirror=True,
            ticks="inside",
            tickcolor="black",
            tickwidth=1,
            showgrid=True,
            gridcolor="#E5E5E5",
            gridwidth=1,
            title_font=dict(size=12, color="black"),
            tickfont=dict(size=10, color="black"),
        )

        fig_pdf.update_layout(
            width=pdf_width,
            height=pdf_height,
            font=dict(family=font_family, color="black"),
            plot_bgcolor="white",
            paper_bgcolor="white",
            showlegend=False,
            margin=dict(t=10, b=40, l=50, r=10),
        )

        pdf_path = pdf_dir / f"{param}.pdf"
        try:
            fig_pdf.write_image(str(pdf_path), format="pdf", engine="kaleido")
        except Exception as e:
            logger.error(
                f"Failed to save PDF for {param}. Make sure 'kaleido' is installed. Error: {e}"
            )

    fig_html.update_layout(
        height=400 * rows if rows > 0 else 400,
        width=1000,
        title_text="",
        font=dict(family=font_family, size=12, color="black"),
        plot_bgcolor="white",
        paper_bgcolor="white",
        showlegend=False,
        margin=dict(t=40, b=60, l=60, r=40),
    )

    fig_html.write_html(str(out_file))
    logger.info(f"Saved combined HTML dashboard: {out_file}")
    logger.info(f"Saved individual PDFs in: {pdf_dir}/")


def generate_distributions():
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    run_dir = project_root / "results" / f"{FOLDER}"

    if not run_dir.exists():
        raise ValueError(f"Directory {run_dir} not found. Run analysis first.")

    logger.info("Extracting metrics and building DataFrames...")
    all_results_list = []

    config_path = run_dir / ".hydra" / "config.yaml"
    stats_path = "stats_multi_pair_*.parquet"
    stats_files = list(run_dir.glob(stats_path))

    if not stats_files:
        stats_files = list(run_dir.rglob(stats_path))

    if not config_path.exists() and not list(run_dir.rglob("config.yaml")):
        logger.warning(f"Config files not found in {run_dir}")

    if not stats_files:
        raise ValueError(f"{stats_path} not found in {run_dir}")

        # Iterujemy po wszystkich znalezionych plikach stats (dla Grid Search!)
    for stats_file in stats_files:
        try:
            # Próba znalezienia powiązanego configu w tym samym folderze lub głównym
            local_config = stats_file.parent / ".hydra" / "config.yaml"
            cfg_to_load = local_config if local_config.exists() else config_path

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
                "Entry": entry,
                "Exit": exit_t,
                "SL": stop_loss,
                "Z-Score Window": z_score_window,
                "Pairs": top_n,
                "CAGR": get_metric("cagr", "net"),
                "Annual Volatility": get_metric("volatility_annual", "net"),
                "Max Drawdown": get_metric("max_drawdown", "net"),
                "Total Trades": int(get_metric("win_count", "net") or 0)
                + int(get_metric("lose_count", "net") or 0),
                "Avg Trade Duration": get_metric("avg_trade_duration", "net"),
                "Sortino Ratio": get_metric("sortino_ratio_annual", "net"),
                "Sortino Ratio Mean": get_metric("sortino_annual_mean", "net"),
                "Sortino Ratio Median": get_metric("sortino_annual_median", "net"),
                "TDA-Sortino": get_metric("tda_sortino", "net"),
            }

            all_results_list.append(row_data)

        except Exception as e:
            logger.error(f"Error processing stats for {stats_file}: {e}")

    logger.info("Saving results and generating plots...")
    df_summary = pd.DataFrame(all_results_list)

    df_summary = df_summary.sort_values(by="TDA-Sortino", ascending=False).reset_index(
        drop=True
    )

    # ==============================================================
    # GENEROWANIE TABELI LATEX (ZANIM ZMIENIMY TYPY NA STRINGI)
    # ==============================================================
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
    tex_path = run_dir / "top_grid_search_results.tex"

    with open(tex_path, "w") as f:
        f.write(
            df_tex.head(TOP_N_ROWS).to_latex(
                index=False, column_format="c" * len(df_tex.columns), escape=False
            )
        )
    logger.info(f"Saved LaTeX table (Top {TOP_N_ROWS}): {tex_path}")

    cols_to_str = [
        "Entry",
        "Exit",
        "SL",
        "Z-Score Window",
        "Pairs",
    ]
    for col in cols_to_str:
        if col in df_summary.columns:
            df_summary[col] = df_summary[col].astype(str)

    output_parquet = run_dir / "summary_distribution.parquet"
    df_summary.to_parquet(output_parquet, engine="pyarrow", index=False)
    logger.info(f"Saved summary: {output_parquet}")

    output_html = run_dir / "plots_distribution.html"
    plot_distribution(
        df_summary,
        title="TDA-Sortino Distribution",
        out_file=output_html,
        params_to_plot=[
            "Entry",
            "Exit",
            "SL",
            "Z-Score Window",
            "Pairs",
        ],
        target_metric="TDA-Sortino",
    )


if __name__ == "__main__":
    generate_distributions()
