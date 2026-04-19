"""Script to generate smoothed PDF plots based on W&B export .csv files."""

import glob
import os
import time
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
import re

from modules.utils.logger import get_logger

logger = get_logger(__name__)

TARGET_SUBFOLDER = "wandb_export"

RENAME_MAP = {
    "recurrent_ppo_autonomous_StepPnLReward_1_0": "1 – StepPnLReward, Autonomous, λ=1.0",
    "recurrent_ppo_autonomous_StepPnLReward_1_2": "2 – StepPnLReward, Autonomous, λ=1.2",
    "recurrent_ppo_standard_StepPnLReward_1_0": "3 – StepPnLReward, Standard, λ=1.0",
    "recurrent_ppo_standard_StepPnLReward_1_2": "4 – StepPnLReward, Standard, λ=1.2",
    "recurrent_ppo_full_StepPnLReward_1_0": "5 – StepPnLReward, Full, λ=1.0",
    "recurrent_ppo_full_StepPnLReward_1_2": "6 – StepPnLReward, Full, λ=1.2",
    "recurrent_ppo_autonomous_TradePnLReward_1_0": "7 – TradePnLReward, Autonomous, λ=1.0",
    "recurrent_ppo_autonomous_TradePnLReward_1_2": "8 – TradePnLReward, Autonomous, λ=1.2",
    "recurrent_ppo_standard_TradePnLReward_1_0": "9 – TradePnLReward, Standard, λ=1.0",
    "recurrent_ppo_standard_TradePnLReward_1_2": "10 – TradePnLReward, Standard, λ=1.2",
    "recurrent_ppo_full_TradePnLReward_1_0": "11 – TradePnLReward, Full, λ=1.0",
    "recurrent_ppo_full_TradePnLReward_1_2": "12 – TradePnLReward, Full, λ=1.2",
    "recurrent_ppo_autonomous_HybridActionReward_1_0": "13 – HybridActionReward, Autonomous, λ=1.0",
    "recurrent_ppo_autonomous_HybridActionReward_1_2": "14 – HybridActionReward, Autonomous, λ=1.2",
    "recurrent_ppo_standard_HybridActionReward_1_0": "15 – HybridActionReward, Standard, λ=1.0",
    "recurrent_ppo_standard_HybridActionReward_1_2": "16 – HybridActionReward, Standard, λ=1.2",
    "recurrent_ppo_full_HybridActionReward_1_0": "17 – HybridActionReward, Full, λ=1.0",
    "recurrent_ppo_full_HybridActionReward_1_2": "18 – HybridActionReward, Full, λ=1.2",
}

ELSEVIER_FONT = "Arial, sans-serif"
FONT_SIZE_TICK = 10
FONT_SIZE_LABEL = 12
FONT_SIZE_TITLE = 13
FONT_SIZE_SUBTITLE = 10
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

PDF_WIDTH = 720
PDF_HEIGHT = 650

axis_style_x = dict(
    showline=True,
    linewidth=1,
    linecolor=COLOR_BLACK,
    mirror=True,
    ticks="inside",
    tickcolor=COLOR_BLACK,
    tickwidth=1,
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

BASE_DIR = Path(__file__).parent.parent / "results" / TARGET_SUBFOLDER


def clean_label(raw_col: str) -> str:
    model_part = raw_col.split(" - ")[0]
    return RENAME_MAP.get(model_part, model_part)


def extract_id_from_label(label: str) -> int:
    match = re.match(r"^(\d+)", label)
    if match:
        return int(match.group(1))
    return 999


def generate_wandb_diagnostics(csv_paths: dict, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_configs = [
        {
            "key": "reward",
            "ylabel": "Mean Reward",
            "title": "Mean Episode Reward",
            "smooth": 0.1,
        },
        {
            "key": "value_loss",
            "ylabel": "Value Loss",
            "title": "Value Loss",
            "smooth": 0.05,
        },
        {
            "key": "pg_loss",
            "ylabel": "PG Loss",
            "title": "Policy Gradient Loss",
            "smooth": 0.05,
        },
        {
            "key": "total_loss",
            "ylabel": "Total Loss",
            "title": "Total Loss",
            "smooth": 0.05,
        },
        {
            "key": "explained_var",
            "ylabel": "Variance Fraction",
            "title": "Explained Variance",
            "smooth": 0.05,
        },
        {
            "key": "avg_equity",
            "ylabel": "Avg Equity",
            "title": "Average Equity",
            "smooth": 0.1,
        },
        {
            "key": "avg_hold",
            "ylabel": "Avg Hold Time",
            "title": "Average Hold Time",
            "smooth": 0.1,
        },
        {
            "key": "avg_win_rate",
            "ylabel": "Win Rate",
            "title": "Average Win Rate",
            "smooth": 0.1,
        },
        {
            "key": "exposure_pct",
            "ylabel": "Exposure %",
            "title": "Exposure Pct",
            "smooth": 0.1,
        },
        {
            "key": "fees_paid",
            "ylabel": "Total Fees",
            "title": "Total Fees Paid",
            "smooth": 0.1,
        },
        {
            "key": "entropy_loss",
            "ylabel": "Entropy Loss",
            "title": "Entropy Loss",
            "smooth": 0.1,
        },
    ]

    for cfg in plot_configs:
        file_path = csv_paths.get(cfg["key"])
        if not file_path or not Path(file_path).exists():
            continue

        df = pd.read_csv(file_path)
        x_col = "Step" if "Step" in df.columns else "global_step"

        data_cols = [
            c
            for c in df.columns
            if " - " in c
            and not c.endswith("__MIN")
            and not c.endswith("__MAX")
            and not c.endswith(" - _step")
        ]

        def sort_key(col_name):
            label = clean_label(col_name)
            return extract_id_from_label(label)

        data_cols = sorted(data_cols, key=sort_key)

        fig = go.Figure()
        added_to_legend = set()

        for col in data_cols:
            valid_data = df[[x_col, col]].dropna()
            steps = valid_data[x_col]
            raw_vals = valid_data[col]

            smoothed_vals = raw_vals.ewm(alpha=cfg["smooth"], adjust=False).mean()

            label = clean_label(col)
            model_id = extract_id_from_label(label)

            color_idx = (
                (model_id - 1) % len(PUBLICATION_COLORS) if model_id != 999 else 0
            )
            color = PUBLICATION_COLORS[color_idx]

            show_leg = label not in added_to_legend
            if show_leg:
                added_to_legend.add(label)

            fig.add_trace(
                go.Scatter(
                    x=steps,
                    y=raw_vals,
                    mode="lines",
                    line=dict(color=color, width=0.5),
                    opacity=0.2,
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=steps,
                    y=smoothed_vals,
                    mode="lines",
                    line=dict(color=color, width=1.5),
                    name=label,
                    showlegend=show_leg,
                )
            )

        fig.update_yaxes(title_text=cfg["ylabel"], **axis_style_y)
        fig.update_xaxes(title_text="Global Training Steps", **axis_style_x)

        alpha_val = cfg["smooth"]
        full_title_text = f"{cfg['title']}<br><span style='font-size:{FONT_SIZE_SUBTITLE}px; color:#555555'>Exponential Moving Average (\u03B1={alpha_val})</span>"

        fig.update_layout(
            showlegend=True,
            width=PDF_WIDTH,
            height=PDF_HEIGHT,
            font=dict(family=ELSEVIER_FONT, size=FONT_SIZE_LABEL, color=COLOR_BLACK),
            plot_bgcolor="white",
            paper_bgcolor="white",
            legend=legend_style,
            margin=dict(t=50, b=180, l=60, r=40),
            title=dict(
                text=full_title_text,
                font=dict(
                    family=ELSEVIER_FONT, size=FONT_SIZE_TITLE, color=COLOR_BLACK
                ),
                x=0.5,
                xanchor="center",
                y=0.96,
            ),
        )

        output_file = output_dir / f"wandb_{cfg['key']}.pdf"

        try:
            fig.write_image(str(output_file), format="pdf")
            logger.info(f"Generated plot: {output_file.name}")
        except Exception as e:
            logger.warning(
                f"Initial save failed for {output_file.name}. Killing kaleido and retrying... {e}"
            )
            os.system("taskkill /F /IM kaleido.exe /T >nul 2>&1")
            time.sleep(1)

            max_retries = 3
            for attempt in range(max_retries):
                try:
                    fig.write_image(str(output_file), format="pdf")
                    logger.info(f"PDF saved on retry {attempt + 1}: {output_file.name}")
                    break
                except Exception as retry_e:
                    if attempt < max_retries - 1:
                        logger.warning(
                            f"Retry {attempt + 1} failed for {output_file.name}. Killing kaleido again..."
                        )
                        os.system("taskkill /F /IM kaleido.exe /T >nul 2>&1")
                        time.sleep(2)
                    else:
                        logger.error(
                            f"Failed to save {output_file.name} after {max_retries} retries: {retry_e}"
                        )


if __name__ == "__main__":
    csv_search_pattern = str(BASE_DIR / "*.csv")
    found_files = glob.glob(csv_search_pattern)

    if not found_files:
        logger.error(f"ERROR: .csv files not found in: {BASE_DIR.absolute()}")
    else:
        csv_files_paths = {}

        metric_identifiers = {
            "ep_rew_mean": "reward",
            "portfolio/total_fees_paid": "fees_paid",
            "portfolio/exposure_pct": "exposure_pct",
            "portfolio/avg_win_rate": "avg_win_rate",
            "portfolio/avg_hold_time": "avg_hold",
            "portfolio/avg_equity": "avg_equity",
            "train/value_loss": "value_loss",
            "train/policy_gradient_loss": "pg_loss",
            "train/entropy_loss": "entropy_loss",
            "train/loss": "total_loss",
            "train/explained_variance": "explained_var",
        }

        for file_path in found_files:
            df = pd.read_csv(file_path, nrows=0)
            columns = df.columns

            for col in columns:
                for identifier, config_key in metric_identifiers.items():
                    if identifier in col:
                        csv_files_paths[config_key] = Path(file_path)
                        break

                if any(identifier in col for identifier in metric_identifiers.keys()):
                    break

        if len(csv_files_paths) > 0:
            generate_wandb_diagnostics(csv_files_paths, BASE_DIR)
        else:
            logger.warning(
                "Found CSV files, but its metrics don't fit the identifiers."
            )

    logger.info("Ending...")
    os._exit(0)
