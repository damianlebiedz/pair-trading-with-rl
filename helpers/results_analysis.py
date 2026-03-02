import yaml
import shutil
import pandas as pd
from pathlib import Path


def parse_results():
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
        print(f"Directory {results_dir} not found")
        return None, None

    for run_dir in list(results_dir.iterdir()):
        if not run_dir.is_dir() or run_dir.name in categories.keys():
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

                categories[category_name].append(
                    {
                        "Run_ID": run_dir.name,
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
                        "Sortino Annual Gross": get_metric(
                            "sortino_ratio_annual", "gross"
                        ),
                        "CAGR Net": get_metric("cagr", "net"),
                        "CAGR Gross": get_metric("cagr", "gross"),
                        "Vol Annual Net": get_metric("volatility_annual", "net"),
                        "Max DD Net": get_metric("max_drawdown", "net"),
                    }
                )

            except Exception as e:
                print(
                    f"Error processing stats for {run_dir.name} in {category_name}: {e}"
                )

    summary_dfs = {}
    for category_name, results_list in categories.items():
        if not results_list:
            continue
        df_summary = pd.DataFrame(results_list)
        df_summary = df_summary.sort_values(
            by="Sortino Annual Net", ascending=False
        ).reset_index(drop=True)
        summary_dfs[category_name] = df_summary

    return summary_dfs, results_dir


if __name__ == "__main__":
    dfs, res_dir = parse_results()
    if dfs and res_dir:
        for cat_name, df in dfs.items():
            output_file = res_dir / cat_name / f"summary_{cat_name}.parquet"
            df.to_parquet(output_file, engine="pyarrow", index=False)
            print(f"--> Saved summary for {cat_name}: {output_file}")
