"""Script to generate a smoothed PDF plot with variance (mean ± std) for 5 seeds."""

import glob
import os
import time
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path

from modules.utils.logger import get_logger

logger = get_logger(__name__)

TARGET_SUBFOLDER = "Wandb Export (sec 5.5.4)"

ELSEVIER_FONT = "Arial, sans-serif"
FONT_SIZE_TICK = 10
FONT_SIZE_LABEL = 12
FONT_SIZE_TITLE = 13
FONT_SIZE_SUBTITLE = 10
COLOR_BLACK = "black"

MAIN_COLOR_HEX = "#1f77b4"
SHADED_COLOR_RGBA = "rgba(31, 119, 180, 0.2)"

PDF_WIDTH = 720
PDF_HEIGHT = 500

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

BASE_DIR = Path(__file__).parent.parent.parent / "results" / TARGET_SUBFOLDER


def generate_variance_plot(csv_path: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    x_col = "Step" if "Step" in df.columns else "global_step"

    reward_cols = [
        c
        for c in df.columns
        if "rollout/ep_rew_mean" in c
        and not c.endswith("__MIN")
        and not c.endswith("__MAX")
    ]

    if not reward_cols:
        logger.error(f"'rollout/ep_rew_mean' columns not found in file {csv_path.name}")
        return

    logger.info(f"Found {len(reward_cols)} seeds for variance plot.")

    df_sub = df[[x_col] + reward_cols].dropna()
    steps = df_sub[x_col]

    alpha = 0.1
    smoothed_data = pd.DataFrame()
    for c in reward_cols:
        smoothed_data[c] = df_sub[c].ewm(alpha=alpha, adjust=False).mean()

    mean_vals = smoothed_data.mean(axis=1)
    std_vals = smoothed_data.std(axis=1)

    upper_vals = mean_vals + std_vals
    lower_vals = mean_vals - std_vals

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=pd.concat([steps, steps[::-1]]),
            y=pd.concat([upper_vals, lower_vals[::-1]]),
            fill="toself",
            fillcolor=SHADED_COLOR_RGBA,
            line=dict(color="rgba(255,255,255,0)"),
            showlegend=True,
            name="± 1 Std Dev",
            hoverinfo="skip",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=steps,
            y=mean_vals,
            mode="lines",
            line=dict(color=MAIN_COLOR_HEX, width=2.5),
            name="Agent 2 (Mean, 5 seeds)",
        )
    )

    fig.update_yaxes(title_text="Mean Reward", **axis_style_y)
    fig.update_xaxes(title_text="Global Training Steps", **axis_style_x)

    full_title_text = f"Mean Episode Reward - Seed Variance Analysis<br><span style='font-size:{FONT_SIZE_SUBTITLE}px; color:#555555'>Exponential Moving Average (\u03B1={alpha}) across 5 independent seeds</span>"

    fig.update_layout(
        showlegend=True,
        width=PDF_WIDTH,
        height=PDF_HEIGHT,
        font=dict(family=ELSEVIER_FONT, size=FONT_SIZE_LABEL, color=COLOR_BLACK),
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend=legend_style,
        margin=dict(t=50, b=80, l=60, r=40),
        title=dict(
            text=full_title_text,
            font=dict(family=ELSEVIER_FONT, size=FONT_SIZE_TITLE, color=COLOR_BLACK),
            x=0.5,
            xanchor="center",
            y=0.96,
        ),
    )

    output_file = output_dir / "wandb_reward_variance.pdf"

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

    target_csv = None
    for file_path in found_files:
        df_cols = pd.read_csv(file_path, nrows=0).columns
        if any("rollout/ep_rew_mean" in c for c in df_cols):
            target_csv = Path(file_path)
            break

    if target_csv:
        generate_variance_plot(target_csv, BASE_DIR)
    else:
        logger.error(f"ERROR: CSV file not found in {BASE_DIR}")

    logger.info("Ending...")
    os._exit(0)
