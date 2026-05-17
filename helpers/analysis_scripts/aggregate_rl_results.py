"""Script to aggregate RL agents performance, including LaTeX table generating."""

import sys
import yaml
import pandas as pd
from pathlib import Path

FOLDER = "RL MODELS OOS 10x"


def get_column_id_from_folder(folder_name: str) -> int:
    """
    Parses the model folder name to determine its column ID (1-18) based on:
    1. Reward Function (Base offset: 0, 6, 12)
    2. State Space (Offset: 0, 2, 4)
    3. Lambda (Offset: 1, 2)
    """
    name_lower = folder_name.lower()

    if "steppnlreward" in name_lower:
        base = 0
    elif "tradepnlreward" in name_lower:
        base = 6
    elif "hybridactionreward" in name_lower:
        base = 12
    else:
        return None

    if "autonomous" in name_lower:
        space_offset = 0
    elif "standard" in name_lower:
        space_offset = 2
    elif "full" in name_lower:
        space_offset = 4
    else:
        return None

    if "1_0" in name_lower:
        lambda_offset = 1
    elif "1_2" in name_lower:
        lambda_offset = 2
    else:
        return None

    return base + space_offset + lambda_offset


def aggregate_models():
    script_dir = Path(__file__).resolve().parent
    base_path = script_dir.parent.parent / "results" / FOLDER

    if not base_path.exists():
        print("Error: The specified folder does not exist.")
        print(f"Python looked for it exactly here: {base_path}")
        sys.exit(1)

    records = []

    for run_dir in base_path.glob("run_backtest*"):
        if not run_dir.is_dir():
            continue

        config_path = run_dir / ".hydra" / "config.yaml"
        rl_model_folder = ""

        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                try:
                    config = yaml.safe_load(f)
                    rl_model_folder = str(config.get("rl_model_folder", ""))
                except yaml.YAMLError as e:
                    print(f"YAML parsing error in {config_path}: {e}")
        else:
            print(f"Skipped {run_dir.name} - missing .hydra/config.yaml")
            continue

        model_col_id = get_column_id_from_folder(rl_model_folder)

        if model_col_id is None:
            print(
                f"Skipped {run_dir.name} - could not parse parameters from '{rl_model_folder}'"
            )
            continue

        stats_files = list(run_dir.rglob("stats_multi_pair_*.parquet"))
        if not stats_files:
            print(f"Skipped {run_dir.name} - missing stats_multi_pair_*.parquet file")
            continue

        stats_file = sorted(stats_files, key=lambda p: len(p.parts))[0]

        try:
            stats_df = pd.read_parquet(stats_file)
        except Exception as e:
            print(f"Error reading parquet file {stats_file}: {e}")
            continue

        if "metric" in stats_df.columns:
            stats_df = stats_df.set_index("metric")
        elif "index" in stats_df.columns:
            stats_df = stats_df.set_index("index")

        if "net" not in stats_df.columns:
            continue

        net_stats = stats_df["net"].to_dict()

        record = {
            "col_id": model_col_id,
            "original_name": rl_model_folder,
            **net_stats,
        }
        records.append(record)

    if not records:
        print("Error: No results found to aggregate.")
        sys.exit(0)

    final_df = pd.DataFrame(records)
    final_df = final_df.sort_values("col_id").reset_index(drop=True)

    output_path = script_dir.parent.parent / "results" / "aggregated_models"
    output_path.mkdir(parents=True, exist_ok=True)

    parquet_path = output_path / "aggregated_stats.parquet"
    tex_path = output_path / "aggregated_stats.tex"

    final_df.drop(columns=["col_id"]).to_parquet(parquet_path, index=False)

    num_models = len(final_df)
    cols_alignment = f"l*{{{num_models}}}{{c}}"
    col_headers = " & " + " & ".join([str(int(i)) for i in final_df["col_id"]])

    def format_row(metric_key, display_name, fmt_type, add_spacing=False):
        if metric_key not in final_df.columns:
            vals = ["-"] * num_models
        else:
            vals = []
            for val in final_df[metric_key]:
                if pd.isna(val):
                    vals.append("-")
                elif fmt_type == "pct":
                    vals.append(f"{val * 100:.2f}\\%")
                elif fmt_type == "int":
                    vals.append(f"{int(val)}")
                elif fmt_type == "float2":
                    vals.append(f"{val:.2f}")
                elif fmt_type == "float4":
                    vals.append(f"{val:.4f}")

        ending = " \\\\[4pt]" if add_spacing else " \\\\"
        return f"    {display_name} & " + " & ".join(vals) + ending

    latex_template = f"""\\begin{{landscape}}
\\vspace*{{\\fill}}
\\renewcommand{{\\arraystretch}}{{1.2}}
\\begin{{center}}
\\footnotesize
\\captionof{{table}}{{Comparison of In-Sample RL Models Performance (2024).}}
\\label{{tab:is_models}}
\\resizebox{{\\linewidth}}{{!}}{{
    \\begin{{tabular}}{{{cols_alignment}}}
    \\hline
{col_headers} \\\\
    \\hline
{format_row('cagr', 'CAGR', 'pct')}
{format_row('volatility_annual', 'Annual Volatility', 'pct')}
{format_row('max_drawdown', 'Max Drawdown', 'pct', add_spacing=True)}

{format_row('win_count', 'Win Count', 'int')}
{format_row('lose_count', 'Loss Count', 'int')}
{format_row('win_rate', 'Win Rate', 'pct', add_spacing=True)}

{format_row('avg_win_return', 'Avg Win Return', 'pct')}
{format_row('avg_lose_return', 'Avg Lose Return', 'pct')}
{format_row('avg_trade_return', 'Avg Trade Return', 'pct')}
{format_row('avg_trade_duration', 'Avg Trade Duration', 'float2', add_spacing=True)}

{format_row('sharpe_ratio_annual', 'Sharpe Ratio (Ann.)', 'float4')}
{format_row('sortino_ratio_annual', 'Sortino Ratio (Ann.)', 'float4')}
{format_row('calmar_ratio', 'Calmar Ratio', 'float4')}
    \\hline
    \\end{{tabular}}
}}
\\scriptsize
\\vspace{{12pt}}

\\justifying \\noindent \\scriptsize 
    Note: The 10x leverage is applied to the agents to scale its inherently lower structural volatility and align its risk profile with the unleveraged benchmarks (see Subsection 2.8). Consequently, all
    calculated performance metrics represent the post-leverage performance of the strategy (see Subsection 2.7). Agents: 1 – StepPnLReward, Autonomous, $\\lambda=1.0$, 2 – StepPnLReward, Autonomous, $\\lambda=1.2$, 3 – StepPnLReward, Standard, $\\lambda=1.0$, 4 – StepPnLReward, Standard, $\\lambda=1.2$, 5 – StepPnLReward, Full, $\\lambda=1.0$, 6 – StepPnLReward, Full, $\\lambda=1.2$, 7 – TradePnLReward, Autonomous, $\\lambda=1.0$, 8 – TradePnLReward, Autonomous, $\\lambda=1.2$, 9 – TradePnLReward, Standard, $\\lambda=1.0$, 10 – TradePnLReward, Standard, $\\lambda=1.2$, 11 – TradePnLReward, Full, $\\lambda=1.0$, 12 – TradePnLReward, Full, $\\lambda=1.2$, 13 – HybridActionReward, Autonomous, $\\lambda=1.0$, 14 – HybridActionReward, Autonomous, $\\lambda=1.2$, 15 – HybridActionReward, Standard, $\\lambda=1.0$, 16 – HybridActionReward, Standard, $\\lambda=1.2$, 17 – HybridActionReward, Full, $\\lambda=1.0$, 18 – HybridActionReward, Full, $\\lambda=1.2$.
\\end{{center}}
\\vspace*{{\\fill}}
\\end{{landscape}}
"""

    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(latex_template)

    print(f"\nSuccess! Aggregated statistics for {len(final_df)} models.")
    print(f"Data saved to:\n  - Parquet: {parquet_path}\n  - LaTeX: {tex_path}")


if __name__ == "__main__":
    aggregate_models()
