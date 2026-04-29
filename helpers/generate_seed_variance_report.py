"""Script to generate Seed Variance performance reports, mirroring the Sensitivity Analysis template logic."""

import os
import warnings
import yaml
import re
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path

from modules.core.enums import Interval
from modules.performance.stats import calculate_stats
from modules.utils.logger import get_logger

warnings.filterwarnings("ignore", category=UserWarning, module="choreographer")
logger = get_logger(__name__)

FOLDER = "RL MODELS OOS 10x SEEDS"
LEVERAGE = 10

TITLE = "Out-Of-Sample Performance Stability Across Random Seeds"

ELSEVIER_FONT = "Arial, sans-serif"
FONT_SIZE_TICK = 10
FONT_SIZE_LABEL = 12
FONT_SIZE_TITLE = 13
COLOR_BLACK = "black"

AGENT_2_SEED = 42
AGENT_2_COLOR = "#FF8C00"

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


def generate_seed_variance_report(folder_name: str):
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    category_dir = project_root / "results" / folder_name

    report_output_dir = category_dir.parent / "seed_variance_report"
    report_output_dir.mkdir(parents=True, exist_ok=True)

    axis_style_x = dict(
        showline=True,
        linewidth=1,
        linecolor=COLOR_BLACK,
        mirror=True,
        ticks="inside",
        tickcolor=COLOR_BLACK,
        tickwidth=1,
        tickangle=0,
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
        tickfont=dict(family=ELSEVIER_FONT, size=FONT_SIZE_TICK, color=COLOR_BLACK),
        title_font=dict(family=ELSEVIER_FONT, size=FONT_SIZE_LABEL, color=COLOR_BLACK),
        showgrid=True,
        gridcolor="#E5E5E5",
        gridwidth=0.5,
        zeroline=True,
        zerolinecolor=COLOR_BLACK,
        zerolinewidth=1,
    )

    def get_run_data(run_dir: Path):
        config_path = run_dir / ".hydra" / "config.yaml"
        ts_files = list(run_dir.glob("returns_multi_pair_*.parquet"))
        exec_files = list(run_dir.glob("exec_logger_*.parquet"))

        if not config_path.exists() or not ts_files:
            return None

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)

            ts_df = pd.read_parquet(ts_files[0])
            exec_df = pd.read_parquet(exec_files[0]) if exec_files else pd.DataFrame()

            def get_cfg(key):
                if key in config:
                    return config[key]
                for section in [
                    "performance",
                    "pair_selection",
                    "market",
                    "settings",
                    "setup",
                ]:
                    if section in config and isinstance(config[section], dict):
                        if key in config[section]:
                            return config[section][key]
                return None

            current_seed = None
            rl_folder = get_cfg("rl_model_folder")

            if rl_folder:
                match = re.search(r"_seed(\d+)", str(rl_folder))
                if match:
                    current_seed = int(match.group(1))

            if current_seed is None:
                current_seed = get_cfg("seed")

            if current_seed is None:
                logger.warning(f"[{run_dir.name}] SKIPPED: Seed info not found.")
                return None

            initial_cash = ts_df["equity"].iloc[0]
            ret_series = (ts_df["equity"] / initial_cash) - 1
            ret_series = ret_series - ret_series.iloc[0]

            stats_files = list(run_dir.glob("stats_*.parquet"))
            if stats_files:
                df_stats = pd.read_parquet(stats_files[0])
                if "metric" in df_stats.columns:
                    df_stats = df_stats.set_index("metric")
                stats_net = df_stats["net"]
            else:
                risk_free_rate = float(get_cfg("risk_free_rate_annual") or 0.0)
                stats_calc = calculate_stats(
                    ts_df, exec_df, initial_cash, Interval.H1, risk_free_rate
                )
                stats_net = stats_calc["net"]

            return {
                "seed": current_seed,
                "metrics": stats_net,
                "ret_series": ret_series,
            }
        except Exception as e:
            logger.error(f"[{run_dir.name}] ERROR: {str(e)}")
            return None

    logger.info(f"Extracting runs from {folder_name}...")
    results = []
    for d in category_dir.iterdir():
        if d.is_dir() and d.name != "seed_variance_report":
            data = get_run_data(d)
            if data:
                results.append(data)

    if not results:
        logger.error("No correct data found.")
        return

    results = sorted(results, key=lambda x: x["seed"])
    seeds = [r["seed"] for r in results]

    # Seed 42 (Agent 2) must always be highlighted in the chosen orange.
    other_seeds = [s for s in seeds if s != AGENT_2_SEED]
    if len(other_seeds) > len(PUBLICATION_COLORS):
        raise ValueError(
            f"Not enough colors configured for seeds: {len(other_seeds)} other seeds "
            f"but only {len(PUBLICATION_COLORS)} colors available."
        )

    # Assign unique colors to non-42 seeds deterministically (sorted by seed).
    other_seed_colors = {s: PUBLICATION_COLORS[i] for i, s in enumerate(other_seeds)}

    fig = go.Figure()
    for res in results:
        series = res["ret_series"]
        seed = res["seed"]
        color = AGENT_2_COLOR if seed == AGENT_2_SEED else other_seed_colors[seed]
        fig.add_trace(
            go.Scatter(
                x=series.index,
                y=series,
                name=f"Seed {seed}",
                line=dict(color=color, width=1.5),
            )
        )

    fig.update_layout(
        width=PDF_WIDTH,
        height=PDF_HEIGHT,
        font=dict(family=ELSEVIER_FONT, size=FONT_SIZE_LABEL, color=COLOR_BLACK),
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(t=30, b=40, l=45, r=10),
        legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5),
        title=dict(
            text=TITLE, font=dict(size=FONT_SIZE_TITLE), x=0.5, xanchor="center"
        ),
    )
    fig.update_xaxes(**axis_style_x, tickformat="%b\n%Y", dtick="M3")
    fig.update_yaxes(**axis_style_y, tickformat=".0%", title_text="Cumulative Return")

    fig.write_image(str(report_output_dir / "seed_variance_equity_oos.pdf"))

    row_groups = [
        [
            ("cagr", "CAGR", "%"),
            ("volatility_annual", "Annual Volatility", "%"),
            ("max_drawdown", "Max Drawdown", "%"),
        ],
        [
            ("win_rate", "Win Rate", "%"),
            ("avg_trade_duration", "Avg Trade Duration", "f"),
        ],
        [
            ("sharpe_ratio_annual", "Sharpe Ratio (Ann.)", "f4"),
            ("sortino_ratio_annual", "Sortino Ratio (Ann.)", "f4"),
            ("calmar_ratio", "Calmar Ratio", "f4"),
        ],
    ]

    def format_val(val, fmt):
        if pd.isnull(val):
            return "-"
        if fmt == "%":
            return f"{val * 100:.2f}\\%"
        if fmt == "f":
            return f"{val:.2f}"
        if fmt == "f4":
            return f"{val:.4f}"
        return str(val)

    total_cols = 1 + len(results) + 2
    col_format = "l" + f"*{{{total_cols - 1}}}{{>{{\\centering\\arraybackslash}}X}}"

    header = (
        "Metric & "
        + " & ".join([f"Seed {s}" for s in seeds])
        + " & Mean & Std Dev \\\\"
    )

    rows_tex = ""
    for group in row_groups:
        for orig_name, tex_name, fmt in group:
            vals = [r["metrics"].get(orig_name) for r in results]
            m, s = np.mean(vals), np.std(vals)

            row = f"        {tex_name}"
            for v in vals:
                row += f" & {format_val(v, fmt)}"
            row += f" & {format_val(m, fmt)} & {format_val(s, fmt)} \\\\"

            if orig_name in ["max_drawdown", "win_rate", "avg_trade_duration"]:
                row += "[4pt]"
            rows_tex += row + "\n"

    tex = f"""\\begin{{table}}[H]
    \\centering\\footnotesize\\renewcommand{{\\arraystretch}}{{1.2}}
    \\caption{{{TITLE}}}
    \\label{{tab:seed-variance-oos}}
    \\begin{{tabularx}}{{\\linewidth}}{{{col_format}}}
    \\toprule {header} \\midrule
{rows_tex.rstrip()}
    \\bottomrule
    \\end{{tabularx}}
    \\vspace{{6pt}}
    \\justifying\\noindent\\scriptsize Note: Performance metrics calculated for the Out-of-Sample period (2025) across 5 independent random seeds. Leverage {LEVERAGE}x.
\\end{{table}}"""

    with open(
        report_output_dir / "seed_variance_table.tex", "w", encoding="utf-8"
    ) as f:
        f.write(tex)

    logger.info("Report generated successfully.")
    os._exit(0)


if __name__ == "__main__":
    generate_seed_variance_report(FOLDER)
