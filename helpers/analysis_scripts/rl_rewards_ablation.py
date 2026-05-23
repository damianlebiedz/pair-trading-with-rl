"""Script to generate PDF equity plots for RL agents grouped by Reward Function."""

import os
import sys
import re
import time
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path

from modules.utils.logger import get_logger

logger = get_logger(__name__)

TARGET_EXPERIMENT_FOLDER = "RL OOS backtests"

ELSEVIER_FONT = "Arial, sans-serif"
FONT_SIZE_TICK = 10
FONT_SIZE_LABEL = 12
FONT_SIZE_TITLE = 13
COLOR_BLACK = "black"

PUBLICATION_COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
    "#393b79",
    "#5254a3",
    "#6b6ecf",
    "#9c9ede",
    "#637939",
    "#8ca252",
    "#b5cf6b",
    "#cedb9c",
    "#8c6d31",
    "#bd9e39",
]

PLOT_SPACE_COLOR_FAMILIES = {
    "StepPnLReward": {
        "autonomous": ["#fdae6b", "#ff7f0e"],
        "standard": ["#9e9ac8", "#756bb1"],
        "full": ["#74c476", "#31a354"],
    },
    "TradePnLReward": {
        "autonomous": ["#8c6d31", "#bd9e39"],
        "standard": ["#e377c2", "#c994c7"],
        "full": ["#17becf", "#9edae5"],
    },
    "HybridActionReward": {
        "autonomous": ["#7f7f7f", "#bdbdbd"],
        "standard": ["#d62728", "#ff9896"],
        "full": ["#1f77b4", "#6baed6"],
    },
}

PDF_WIDTH = 720
PDF_HEIGHT = 400

current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
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

    report_output_dir = project_root / "results" / "rl_rewards_ablation"
    report_output_dir.mkdir(parents=True, exist_ok=True)

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
        entrywidth=240,
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

        line_label = f"{col_id} - {reward}, {space.capitalize()}, λ={lam_label}"

        initial_cash = df_ret["equity"].iloc[0]
        ret = (df_ret["equity"] / initial_cash) - 1
        ret = ret - ret.iloc[0]

        grouped_series[reward].append(
            {
                "col_id": col_id,
                "space": space,
                "series": ret,
                "label_full": line_label,
            }
        )

    logger.info("Generating PDF plots...")

    # Build a global color map with plot-specific families:
    # similar shades within the same space *inside one plot*, but different families across plots.
    all_series = [
        item for series_list in grouped_series.values() for item in series_list
    ]
    all_col_ids = sorted({item["col_id"] for item in all_series})
    color_map = {}

    for reward_type, reward_families in PLOT_SPACE_COLOR_FAMILIES.items():
        reward_items = grouped_series.get(reward_type, [])
        reward_col_ids = {item["col_id"] for item in reward_items}

        for space_name, family_colors in reward_families.items():
            space_ids = sorted(
                {item["col_id"] for item in reward_items if item["space"] == space_name}
            )

            if len(space_ids) > len(family_colors):
                raise ValueError(
                    f"Not enough colors configured for plot '{reward_type}' and space '{space_name}': "
                    f"{len(space_ids)} series but only {len(family_colors)} colors configured."
                )

            for idx, cid in enumerate(space_ids):
                color_map[cid] = family_colors[idx]

        unassigned_ids = sorted(cid for cid in reward_col_ids if cid not in color_map)
        if unassigned_ids:
            raise ValueError(
                f"Missing colors for plot '{reward_type}' and col_id values: {unassigned_ids}"
            )

    missing_ids = [cid for cid in all_col_ids if cid not in color_map]
    if missing_ids:
        raise ValueError(f"Missing colors for col_id values: {missing_ids}")

    # Keep Agent 2 on the same orange highlight color as before.
    color_map[2] = "#ff7f0e"

    for reward_type, series_list in grouped_series.items():
        if not series_list:
            continue

        fig = go.Figure()

        series_list.sort(key=lambda x: x["col_id"])

        for data in series_list:
            current_color = color_map[data["col_id"]]

            fig.add_trace(
                go.Scatter(
                    x=data["series"].index,
                    y=data["series"],
                    name=data["label_full"],
                    line=dict(color=current_color, width=1.5),
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

        pdf_path = report_output_dir / f"comparison_{reward_type}.pdf"
        try:
            fig.write_image(str(pdf_path), format="pdf")
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
                    fig.write_image(str(pdf_path), format="pdf")
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
        logger.info(f"Saved: {pdf_path.name}")

    logger.info(
        f"Plot generation completed successfully! Saved to: {report_output_dir}"
    )

    logger.info("Ending...")
    os._exit(0)


if __name__ == "__main__":
    generate_multi_report(TARGET_EXPERIMENT_FOLDER)
