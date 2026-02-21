import re
import math
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

PATH = "SL_GRID_5_hybrid_fixed"


def extract_param(param_name, text):
    match = re.search(rf"{param_name}:\s*(\S+)", text)
    if match:
        return match.group(1).replace("'", "").replace('"', "")
    return "N/A"


def create_html_plot(df: pd.DataFrame, title: str, out_file: Path):
    params_to_plot = ["Delayed Entry", "SL Lock", "Time Decay SL"]
    valid_params = [
        p for p in params_to_plot if p in df.columns and df[p].nunique() > 1
    ]

    if not valid_params:
        print(
            f"\n--> No variable parameters to plot for: {title} (all have a single value)."
        )
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
            df[f"{param}_str"].unique(),
            key=lambda x: (x.lower() in ["null", "none"], x),
        )

        fig.add_trace(
            go.Box(
                x=df[f"{param}_str"],
                y=df["Sortino Annual Net"],
                text=df["Run_ID"],
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
        width=1300,
        title_text=f"Sortino Distribution Analysis: <b>{title}</b>",
        title_font_size=20,
        plot_bgcolor="white",
        paper_bgcolor="white",
        hovermode="closest",
    )

    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="#E5E5E5")
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="#E5E5E5")

    fig.write_html(str(out_file))
    print(f"--> Saved interactive HTML dashboard: {out_file}")


def summarize_folder(folder_name):
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    target_dir = project_root / "results" / folder_name

    if not target_dir.exists() or not target_dir.is_dir():
        print(f" Error: Directory '{target_dir}' does not exist.")
        return

    results_list = []
    print(f"Searching directory: {target_dir}...")

    for run_dir in target_dir.iterdir():
        if not run_dir.is_dir():
            continue

        log_file = run_dir / "execution.log"
        stats_files = list(run_dir.glob("stats_multi_pair_*.parquet"))

        if not log_file.exists() or not stats_files:
            continue

        stats_file = stats_files[0]

        with open(log_file, "r", encoding="utf-8") as f:
            content = f.read()

        delayed_entry = extract_param("delayed_entry", content)
        sl_lock = extract_param("sl_lock", content)
        time_decay_sl = extract_param("time_decay_sl", content)

        try:
            stats_df = pd.read_parquet(stats_file)

            def get_metric(metric_name, col_type):
                row = stats_df[stats_df["metric"] == metric_name]
                if not row.empty:
                    return row[col_type].iloc[0]
                return None

            cagr_net = get_metric("cagr", "net")
            volatility_annual_net = get_metric("volatility_annual", "net")
            sortino_annual_net = get_metric("sortino_ratio_annual", "net")
            max_dd_net = get_metric("max_drawdown", "net")

            cagr_gross = get_metric("cagr", "gross")
            volatility_annual_gross = get_metric("volatility_annual", "gross")
            sortino_annual_gross = get_metric("sortino_ratio_annual", "gross")
            max_dd_gross = get_metric("max_drawdown", "gross")

            win_count = get_metric("win_count", "net")
            lose_count = get_metric("lose_count", "net")
            total_trades = (win_count if pd.notna(win_count) else 0) + (
                lose_count if pd.notna(lose_count) else 0
            )

            results_list.append(
                {
                    "Run_ID": run_dir.name,
                    "Delayed Entry": delayed_entry,
                    "SL Lock": sl_lock,
                    "Time Decay SL": time_decay_sl,
                    "Total Trades": int(total_trades),
                    "Sortino Annual Net": sortino_annual_net,
                    "Sortino Annual Gross": sortino_annual_gross,
                    "CAGR Net": cagr_net,
                    "CAGR Gross": cagr_gross,
                    "Vol Annual Net": volatility_annual_net,
                    "Vol Annual Gross": volatility_annual_gross,
                    "Max DD Net": max_dd_net,
                    "Max DD Gross": max_dd_gross,
                }
            )
        except Exception as e:
            print(f" Error processing {run_dir.name}: {e}")

    if not results_list:
        print(f" No valid results found in '{folder_name}'.")
        return

    df_summary = pd.DataFrame(results_list)
    df_summary["Sortino Annual Net"] = pd.to_numeric(
        df_summary["Sortino Annual Net"], errors="coerce"
    )
    df_summary["Total Trades"] = pd.to_numeric(
        df_summary["Total Trades"], errors="coerce"
    )

    df_summary = df_summary.sort_values(
        by="Sortino Annual Net", ascending=False
    ).reset_index(drop=True)

    out_parquet = target_dir / f"summary_{folder_name}.parquet"
    df_summary.to_parquet(out_parquet, engine="pyarrow", index=False)

    print(f"\n--> Finished! Found {len(df_summary)} simulations.")
    print(f"--> Saved Parquet: {out_parquet}")

    out_html = target_dir / f"plots_sortino_{folder_name}.html"
    create_html_plot(df_summary, title=folder_name, out_file=out_html)


if __name__ == "__main__":
    summarize_folder(PATH)
