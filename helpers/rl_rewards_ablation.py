"""Script to generate PDF equity plots for RL agents grouped by Reward Function."""

import sys
import re
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
from datetime import datetime

from modules.utils.logger import get_logger

logger = get_logger(__name__)

TARGET_EXPERIMENT_FOLDER = "RL OOS"

LEVERAGE = 10
ELSEVIER_FONT = "Arial, sans-serif"
FONT_SIZE_TICK = 10
FONT_SIZE_LABEL = 12
FONT_SIZE_TITLE = 13
COLOR_BLACK = "black"

PUBLICATION_COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b"]

PDF_WIDTH = 720
PDF_HEIGHT = 400

current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
sys.path.append(str(project_root))

LEGEND_MAP = {
    ("StepPnLReward", "autonomous", "1_0"): 1,
    ("StepPnLReward", "autonomous", "1_2"): 2,
    ("StepPnLReward", "standard", "1_0"): 3,
    ("StepPnLReward", "standard", "1_2"): 4,
    ("StepPnLReward", "full", "1_0"): 5,
    ("StepPnLReward", "full", "1_2"): 6,
    ("TradePnLReward", "autonomous", "1_0"): 7,
    ("TradePnLReward", "autonomous", "1_2"): 8,
    ("TradePnLReward", "standard", "1_0"): 9,
    ("TradePnLReward", "standard", "1_2"): 10,
    ("TradePnLReward", "full", "1_0"): 11,
    ("TradePnLReward", "full", "1_2"): 12,
    ("HybridActionReward", "autonomous", "1_0"): 13,
    ("HybridActionReward", "autonomous", "1_2"): 14,
    ("HybridActionReward", "standard", "1_0"): 15,
    ("HybridActionReward", "standard", "1_2"): 16,
    ("HybridActionReward", "full", "1_0"): 17,
    ("HybridActionReward", "full", "1_2"): 18,
}


def load_returns_data(strat_dir: Path):
    if not strat_dir.exists():
        return None
    returns_files = list(strat_dir.glob("returns_*.parquet"))
    return pd.read_parquet(returns_files[0]) if returns_files else None


def get_rl_model_folder(strat_dir: Path) -> str:
    config_path = strat_dir / ".hydra" / "config.yaml"
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            content = f.read()
            match = re.search(r"rl_model_folder:\s*([^\n]+)", content)
            if match:
                return match.group(1).strip()
    return ""


def parse_model_folder(folder_string: str):
    match = re.search(
        r"(autonomous|standard|full)_(StepPnLReward|TradePnLReward|HybridActionReward)_(1_0|1_2)",
        folder_string,
    )
    if match:
        return match.group(2), match.group(1), match.group(3)
    return None, None, None


def generate_multi_report(target_folder_name: str):
    base_dir = project_root / "results" / target_folder_name
    if not base_dir.exists():
        logger.error(f"Directory {base_dir} does not exist!")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_output_dir = project_root / "results" / f"final_grouped_plots_{timestamp}"
    pdf_dir = report_output_dir / "pdfs"
    report_output_dir.mkdir(parents=True, exist_ok=True)
    pdf_dir.mkdir(parents=True, exist_ok=True)

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
        y=-0.18,
        xanchor="center",
        x=0.5,
        entrywidth=220,
        entrywidthmode="pixels",
        font=dict(family=ELSEVIER_FONT, size=FONT_SIZE_TICK, color=COLOR_BLACK),
        bgcolor="rgba(255, 255, 255, 0)",
        borderwidth=0,
    )

    grouped_series = {
        "StepPnLReward": [],
        "TradePnLReward": [],
        "HybridActionReward": [],
    }

    initial_cash = 100000

    logger.info("Loading return series...")

    for strat_dir in base_dir.glob("run_backtest*"):
        if not strat_dir.is_dir():
            continue

        model_str = get_rl_model_folder(strat_dir)
        if not model_str:
            continue

        reward, space, lam = parse_model_folder(model_str)
        if not reward:
            continue

        col_id = LEGEND_MAP.get((reward, space, lam))
        if col_id is None:
            continue

        df_ret = load_returns_data(strat_dir)
        if df_ret is None or df_ret.empty:
            continue

        lam_label = "1.2" if lam == "1_2" else "1.0"
        line_label = f"{reward}, {space.capitalize()}, λ={lam_label}"

        pnl = (df_ret["total_pnl"] - df_ret["total_fees"]) * LEVERAGE
        ret = pnl / initial_cash
        ret = ret - ret.iloc[0]

        grouped_series[reward].append(
            {"col_id": col_id, "series": ret, "label_full": line_label}
        )

    logger.info("Generating PDF plots...")

    for reward_type, series_list in grouped_series.items():
        if not series_list:
            continue

        fig = go.Figure()

        # Sort lines by col_id so colors are consistent across plots
        series_list.sort(key=lambda x: x["col_id"])

        for i, data in enumerate(series_list):
            fig.add_trace(
                go.Scatter(
                    x=data["series"].index,
                    y=data["series"],
                    name=data["label_full"],
                    line=dict(
                        color=PUBLICATION_COLORS[i % len(PUBLICATION_COLORS)], width=1.5
                    ),
                )
            )

        fig.update_layout(
            width=PDF_WIDTH,
            height=PDF_HEIGHT,
            font=dict(family=ELSEVIER_FONT, size=FONT_SIZE_LABEL, color=COLOR_BLACK),
            plot_bgcolor="white",
            paper_bgcolor="white",
            legend=legend_style,
            margin=dict(t=50, b=110, l=60, r=30),
            title=dict(
                text=f"Out-Of-Sample Performance (2025) - {reward_type}",
                font=dict(size=FONT_SIZE_TITLE),
                x=0.5,
                xanchor="center",
            ),
        )

        fig.update_xaxes(**axis_style_x, tickformat="%b\n%Y")
        fig.update_yaxes(
            **axis_style_y, tickformat=".0%", title_text="Cumulative Return"
        )

        pdf_path = pdf_dir / f"comparison_{reward_type}.pdf"
        fig.write_image(str(pdf_path), format="pdf")
        logger.info(f"Saved: {pdf_path.name}")

    logger.info(f"Plot generation completed successfully! Saved to: {pdf_dir}")


if __name__ == "__main__":
    generate_multi_report(TARGET_EXPERIMENT_FOLDER)
