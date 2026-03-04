import yaml
import shutil
import math
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path


def plot_distribution(df: pd.DataFrame, title: str, out_file: Path):
    params_to_plot = [
        "Category",
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
        df[f"{param}_str"] = df[f"{param}_str"].replace(
            {"nan": "None", "NaN": "None", "<NA>": "None"}
        )
        unique_vals = sorted(
            df[f"{param}_str"].unique(), key=lambda x: (x.lower() == "none", x)
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

        medians = (
            df.groupby(f"{param}_str")["Sortino Annual Net"]
            .median()
            .reindex(unique_vals)
        )

        show_legend_flag = True if i == 0 else False

        fig.add_trace(
            go.Scatter(
                x=unique_vals,
                y=medians.values,
                mode="markers",
                marker=dict(symbol="line-ew", size=40, line=dict(color="red", width=3)),
                hoverinfo="y+name",
                name="Median",
                legendgroup="median_group",
                showlegend=show_legend_flag,
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
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="#E5E5E5")
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="#E5E5E5")

    fig.write_html(str(out_file))
    print(f"--> Saved dashboard: {out_file}")


def process_all_results():
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    results_dir = project_root / "results"

    categories = {
        "1_no_hedge": [],
        "2_static": [],
        "3_rolling": [],
        "0_other": [],
    }

    if not results_dir.exists():
        print(f"Directory {results_dir} not found. Run analysis first.")
        return

    print("Categorizing results...")
    for run_dir in list(results_dir.iterdir()):
        if (
            not run_dir.is_dir()
            or run_dir.name in categories.keys()
            or run_dir.name == "report"
        ):
            continue

        config_path = run_dir / ".hydra" / "config.yaml"
        if not config_path.exists():
            continue

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)

            perf = config.get("performance", {})
            beta_hedge = perf.get("beta_hedge", "N/A")

            if beta_hedge == "no_hedge":
                category_name = "1_no_hedge"
            elif beta_hedge == "static":
                category_name = "2_static"
            elif beta_hedge == "rolling":
                category_name = "3_rolling"
            else:
                category_name = "0_other"

            target_dir = results_dir / category_name
            target_dir.mkdir(exist_ok=True)

            new_run_dir = target_dir / run_dir.name
            shutil.move(str(run_dir), str(new_run_dir))

        except Exception as e:
            print(f"Error categorizing {run_dir.name}: {e}")

    print("\nExtracting metrics and building DataFrames...")
    all_results_list = []

    for category_name in categories.keys():
        cat_dir = results_dir / category_name
        if not cat_dir.exists():
            continue

        for run_dir in cat_dir.iterdir():
            if not run_dir.is_dir():
                continue

            config_path = run_dir / ".hydra" / "config.yaml"
            stats_files = list(run_dir.glob("stats_multi_pair_*.parquet"))

            if not config_path.exists() or not stats_files:
                continue

            stats_file = stats_files[0]

            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    config = yaml.safe_load(f)

                perf = config.get("performance", {})
                ps = config.get("pair_selection", {})

                entry = config.get("entry_threshold", "N/A")
                exit_t = config.get("exit_threshold", "N/A")
                stop_loss = config.get("stop_loss", "N/A")
                z_score_window = config.get("z_score_window", "N/A")
                top_n_factor = ps.get("top_n_factor", "N/A")
                beta_hedge = perf.get("beta_hedge", "N/A")

                stats_df = pd.read_parquet(stats_file)

                def get_metric(metric_name, col_type):
                    row = stats_df[stats_df["metric"] == metric_name]
                    if not row.empty:
                        return row[col_type].iloc[0]
                    return None

                row_data = {
                    "Run_ID": run_dir.name,
                    "Category": category_name,
                    "Pairs": top_n_factor,
                    "Beta Hedge": beta_hedge,
                    "Z-Score Window": z_score_window,
                    "Entry": entry,
                    "Exit": exit_t,
                    "SL": stop_loss,
                    "Total Trades": int(
                        (get_metric("win_count", "net") or 0)
                        + (get_metric("lose_count", "net") or 0)
                    ),
                    "Sortino Annual Net": get_metric("sortino_ratio_annual", "net"),
                    "Sortino Annual Gross": get_metric("sortino_ratio_annual", "gross"),
                    "CAGR Net": get_metric("cagr", "net"),
                    "CAGR Gross": get_metric("cagr", "gross"),
                    "Vol Annual Net": get_metric("volatility_annual", "net"),
                    "Max DD Net": get_metric("max_drawdown", "net"),
                }

                categories[category_name].append(row_data)
                all_results_list.append(row_data)

            except Exception as e:
                print(
                    f"Error processing stats for {run_dir.name} in {category_name}: {e}"
                )

    print("\nSaving results and generating plots...")
    for category_name, results_list in categories.items():
        if not results_list:
            continue

        df_summary = pd.DataFrame(results_list)

        df_summary = df_summary[df_summary["Sortino Annual Net"] >= 0]
        if df_summary.empty:
            print(
                f"--> All strategies in {category_name} with Sortino Net Annual < 0. Skipping."
            )
            continue

        df_summary = df_summary.sort_values(
            by="Sortino Annual Net", ascending=False
        ).reset_index(drop=True)

        cat_dir = results_dir / category_name

        output_parquet = cat_dir / f"summary_{category_name}.parquet"
        df_summary.to_parquet(output_parquet, engine="pyarrow", index=False)
        print(f"--> Saved LOCAL summary: {output_parquet}")

        output_html = cat_dir / f"plots_sortino_{category_name}.html"
        plot_distribution(df_summary, title=category_name, out_file=output_html)

    if all_results_list:
        print("\n--- Generating GLOBAL DataFrame ---")
        global_df = pd.DataFrame(all_results_list)

        global_df = global_df[global_df["Sortino Annual Net"] >= 0]

        if not global_df.empty:
            global_df = global_df.sort_values(
                by="Sortino Annual Net", ascending=False
            ).reset_index(drop=True)

            global_parquet = results_dir / "summary_GLOBAL.parquet"
            global_df.to_parquet(global_parquet, engine="pyarrow", index=False)
            print(f"--> Saved GLOBAL summary: {global_parquet}")

            global_html = results_dir / "plots_sortino_GLOBAL.html"
            plot_distribution(global_df, title="ALL STRATEGIES", out_file=global_html)
        else:
            print("\nStrategies with > 0 Sortino Net Annual not found.")
    else:
        print("\nNo data available to generate global plot.")


if __name__ == "__main__":
    process_all_results()
