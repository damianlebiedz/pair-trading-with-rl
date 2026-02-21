import re
import ast
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

# ==========================================
PATH = "opt_experiments"
# ==========================================


def parse_bounds(content: str) -> dict:
    bounds = {}
    patterns = {
        "entry_threshold_min": r"entry_threshold_min:\s*([\d\.]+)",
        "entry_threshold_max": r"entry_threshold_max:\s*([\d\.]+)",
        "exit_threshold_min": r"exit_threshold_min:\s*([\d\.]+)",
        "exit_threshold_max": r"exit_threshold_max:\s*([\d\.]+)",
        "stop_loss_min": r"stop_loss_min:\s*([\d\.]+)",
        "stop_loss_max": r"stop_loss_max:\s*([\d\.]+)",
        "window_min": r"window_min:\s*([\d\.]+)",
        "window_max": r"window_max:\s*([\d\.]+)",
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, content)
        if match:
            bounds[key] = float(match.group(1))

    return bounds


def parse_iterations(content: str) -> list:
    iterations = []

    for line in content.split("\n"):
        if "INFO - ({" in line and "fixed_window" in line:
            match = re.search(r"INFO - \((\{.*?}),", line)
            if match:
                dict_str = match.group(1)
                try:
                    params = ast.literal_eval(dict_str)
                    iterations.append(params)
                except Exception as e:
                    print(f" Warning: Could not parse dict: {dict_str} | Error: {e}")

    return iterations


def create_optimization_dashboard(
    df: pd.DataFrame, bounds: dict, title: str, out_file: Path
):
    param_config = {
        "fixed_window": {
            "min": "window_min",
            "max": "window_max",
            "title": "Fixed Window",
        },
        "entry_threshold": {
            "min": "entry_threshold_min",
            "max": "entry_threshold_max",
            "title": "Entry Threshold",
        },
        "exit_threshold": {
            "min": "exit_threshold_min",
            "max": "exit_threshold_max",
            "title": "Exit Threshold",
        },
        "stop_loss": {
            "min": "stop_loss_min",
            "max": "stop_loss_max",
            "title": "Stop Loss",
        },
    }

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[config["title"] for config in param_config.values()],
        vertical_spacing=0.15,
        horizontal_spacing=0.1,
    )

    positions = [(1, 1), (1, 2), (2, 1), (2, 2)]

    for (param_col, config), (r, c) in zip(param_config.items(), positions):
        if param_col not in df.columns:
            continue

        b_min = bounds.get(config["min"], 0)
        b_max = bounds.get(config["max"], 0)

        fig.add_trace(
            go.Scatter(
                x=df["Iteration"],
                y=df[param_col],
                mode="markers+lines",
                marker=dict(
                    size=10, color="#1f77b4", line=dict(width=1, color="DarkSlateGrey")
                ),
                line=dict(color="rgba(31, 119, 180, 0.4)", width=2, dash="dot"),
                name=config["title"],
                showlegend=False,
                hovertemplate="Iter: %{x}<br>Value: %{y:.3f}<extra></extra>",
            ),
            row=r,
            col=c,
        )

        fig.add_hline(
            y=b_max,
            line_dash="dash",
            line_color="red",
            opacity=0.7,
            annotation_text="Max Limit",
            annotation_position="top right",
            row=r,
            col=c,
        )
        fig.add_hline(
            y=b_min,
            line_dash="dash",
            line_color="green",
            opacity=0.7,
            annotation_text="Min Limit",
            annotation_position="bottom right",
            row=r,
            col=c,
        )

        padding = (b_max - b_min) * 0.1 if b_max > b_min else 1.0
        fig.update_yaxes(
            range=[b_min - padding, b_max + padding],
            title_text="Parameter Value",
            row=r,
            col=c,
        )
        fig.update_xaxes(
            title_text="Iteration Number",
            tickmode="linear",
            tick0=1,
            dtick=1,
            row=r,
            col=c,
        )

    fig.update_layout(
        height=800,
        width=1200,
        title_text=f"Optimization Parameter Spread: <b>{title}</b>",
        title_font_size=20,
        plot_bgcolor="white",
        paper_bgcolor="white",
        hovermode="x unified",
    )

    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="#E5E5E5")
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="#E5E5E5")

    fig.write_html(str(out_file))
    print(f"--> Saved Optimization Dashboard: {out_file}")


def summarize_optimization_folder(folder_name: str):
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    target_dir = project_root / "results" / folder_name

    if not target_dir.exists() or not target_dir.is_dir():
        print(f" Error: Directory '{target_dir}' does not exist.")
        return

    print(f"Scanning optimization directory: {target_dir}...")

    for run_dir in target_dir.iterdir():
        if not run_dir.is_dir():
            continue

        log_file = run_dir / "execution.log"

        if not log_file.exists():
            continue

        with open(log_file, "r", encoding="utf-8") as f:
            content = f.read()

        bounds = parse_bounds(content)
        iterations = parse_iterations(content)

        if not iterations:
            print(f" Warning: No optimized parameters found in {run_dir.name}.")
            continue

        df = pd.DataFrame(iterations)
        df.insert(0, "Iteration", range(1, len(df) + 1))

        print(f" Processed {run_dir.name} | Found {len(iterations)} iterations.")

        out_html = run_dir / f"plot_optimization_{run_dir.name}.html"
        create_optimization_dashboard(df, bounds, title=run_dir.name, out_file=out_html)


if __name__ == "__main__":
    summarize_optimization_folder(PATH)
