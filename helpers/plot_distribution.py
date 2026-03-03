import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import math


def plot_distribution(df: pd.DataFrame, title: str, out_file: Path):
    params_to_plot = [
        "Category",
        "Entry",
        "Exit",
        "SL",
        "Z-Score Window",
        "Beta Hedge",
        "Pairs",
        "Freeze Std",
    ]

    valid_params = [
        p for p in params_to_plot if p in df.columns and df[p].nunique() > 1
    ]

    if not valid_params:
        print(f" No variable parameters to plot for: {title}.")
        return

    n_params = len(valid_params)
    cols = 2
    rows = math.ceil(n_params / cols)

    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=[f"Distribution by {p}" for p in valid_params],
        vertical_spacing=0.1,
    )

    for i, param in enumerate(valid_params):
        r = (i // cols) + 1
        c = (i % cols) + 1

        df[f"{param}_str"] = df[param].astype(str)
        unique_vals = sorted(
            df[f"{param}_str"].unique(), key=lambda x: (x.lower() == "null", x)
        )

        fig.add_trace(
            go.Box(
                x=df[f"{param}_str"],
                y=df["Sortino Annual Net"],
                text=df["Run_ID"] + "<br>Cat: " + df.get("Category", ""),
                hoverinfo="y+text",
                boxpoints="all",
                jitter=0.5,
                pointpos=0,
                fillcolor="rgba(0,0,0,0)",
                line=dict(color="rgba(0,0,0,0)"),
                marker=dict(
                    size=8,
                    opacity=0.6,
                    color="#1f77b4",
                    line=dict(width=1, color="DarkSlateGrey"),
                ),
                name=param,
                showlegend=False,
            ),
            row=r,
            col=c,
        )

        means = (
            df.groupby(f"{param}_str")["Sortino Annual Net"].mean().reindex(unique_vals)
        )

        fig.add_trace(
            go.Scatter(
                x=unique_vals,
                y=means.values,
                mode="markers",
                marker=dict(symbol="line-ew", size=40, color="red", line_width=4),
                hoverinfo="skip",
                showlegend=False,
            ),
            row=r,
            col=c,
        )

        fig.add_hline(
            y=0, line_dash="dash", line_color="black", opacity=0.5, row=r, col=c
        )

        fig.update_xaxes(
            categoryorder="array",
            categoryarray=unique_vals,
            title_text=param,
            row=r,
            col=c,
        )
        fig.update_yaxes(title_text="Sortino Annual Net", row=r, col=c)

    fig.update_layout(
        height=450 * rows,
        width=1400,
        title_text=f"Sortino Distribution Analysis: <b>{title}</b>",
        title_font_size=22,
        plot_bgcolor="white",
        paper_bgcolor="white",
        hovermode="closest",
    )

    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="#E5E5E5")
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="#E5E5E5")

    fig.write_html(str(out_file))
    print(f"--> Saved dashboard: {out_file}")


def generate_all_plots():
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    results_dir = project_root / "results"

    if not results_dir.exists():
        print(f"Directory {results_dir} does not exist. Run analysis first.")
        return

    all_dfs = []

    for cat_dir in results_dir.iterdir():
        if not cat_dir.is_dir():
            continue

        summary_files = list(cat_dir.glob("summary_*.parquet"))
        if not summary_files:
            continue

        summary_file = summary_files[0]

        try:
            df = pd.read_parquet(summary_file)
        except Exception as e:
            print(f" Error reading {summary_file}: {e}")
            continue

        if df.empty:
            continue

        df["Category"] = cat_dir.name
        all_dfs.append(df)

        print(f"\nGenerating LOCAL plots for: {cat_dir.name}...")
        local_out = cat_dir / f"plots_sortino_{cat_dir.name}.html"
        plot_distribution(df, title=cat_dir.name, out_file=local_out)

    if all_dfs:
        print("Generating GLOBAL plot for all simulations...")
        global_df = pd.concat(all_dfs, ignore_index=True)
        global_out = results_dir / "plots_sortino_GLOBAL.html"
        plot_distribution(global_df, title="ALL STRATEGIES", out_file=global_out)
    else:
        print("\nNo data available to generate global plot.")


if __name__ == "__main__":
    generate_all_plots()
