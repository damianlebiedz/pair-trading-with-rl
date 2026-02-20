import re
import shutil
import pandas as pd
from pathlib import Path


def parse_results():
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    results_dir = project_root / "results"

    categories = {
        "1_baseline": [],
        "2_static": [],
        "3_rolling": [],
        "4_hybrid_no_hedge": [],
        "5_hybrid_fixed": [],
        "0_other": [],
    }

    if not results_dir.exists():
        print(f"Directory {results_dir} not found")
        return None, None

    def extract_param(param_name, text):
        match = re.search(rf"{param_name}:\s*(\S+)", text)
        if match:
            return match.group(1).replace("'", "").replace('"', "")
        return "N/A"

    for run_dir in list(results_dir.iterdir()):
        if not run_dir.is_dir():
            continue

        if run_dir.name in categories.keys():
            continue

        log_file = run_dir / "execution.log"

        if not log_file.exists():
            continue

        with open(log_file, "r", encoding="utf-8") as f:
            content = f.read()

        beta_hedge = extract_param("beta_hedge", content)
        window_method = extract_param("window_method", content)

        if beta_hedge == "no_hedge" and window_method == "fixed":
            category_name = "1_baseline"
        elif beta_hedge == "static" and window_method == "static":
            category_name = "2_static"
        elif beta_hedge == "rolling" and window_method == "rolling":
            category_name = "3_rolling"
        elif beta_hedge == "no_hedge":
            category_name = "4_hybrid_no_hedge"
        elif window_method == "fixed":
            category_name = "5_hybrid_fixed"
        else:
            category_name = "0_other"

        target_dir = results_dir / category_name
        target_dir.mkdir(exist_ok=True)

        new_run_dir = results_dir / category_name / run_dir.name
        shutil.move(str(run_dir), str(new_run_dir))

    for category_name in categories.keys():
        cat_dir = results_dir / category_name

        if not cat_dir.exists():
            continue

        for run_dir in cat_dir.iterdir():
            if not run_dir.is_dir():
                continue

            log_file = run_dir / "execution.log"
            stats_files = list(run_dir.glob("stats_multi_pair_*.parquet"))

            if not log_file.exists() or not stats_files:
                continue

            stats_file = stats_files[0]

            with open(log_file, "r", encoding="utf-8") as f:
                content = f.read()

            entry = extract_param("entry_threshold", content)
            exit_t = extract_param("exit_threshold", content)
            stop_loss = extract_param("stop_loss", content)
            fixed_window = extract_param("fixed_window", content)
            beta_method = extract_param("beta_method", content)
            beta_hedge = extract_param("beta_hedge", content)
            window_method = extract_param("window_method", content)

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

                categories[category_name].append(
                    {
                        "Run_ID": run_dir.name,
                        "Beta Method": beta_method,
                        "Beta Hedge": beta_hedge,
                        "Window Method": window_method,
                        "Fixed Window": fixed_window,
                        "Entry": entry,
                        "Exit": exit_t,
                        "SL": stop_loss,
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
                print(f"Error for {run_dir.name} in {category_name}: {e}")

    summary_dfs = {}
    for category_name, results_list in categories.items():
        if not results_list:
            continue

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
        summary_dfs[category_name] = df_summary

    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)

    return summary_dfs, results_dir


if __name__ == "__main__":
    dfs, res_dir = parse_results()

    if dfs and res_dir:
        for cat_name, df in dfs.items():
            if not df.empty:
                output_file = res_dir / cat_name / f"summary_{cat_name}.parquet"
                df.to_parquet(output_file, engine="pyarrow", index=False)
                print(f"\n--> Saved summary for {cat_name}: {output_file}")
