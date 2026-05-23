"""Post-hoc correlation and VIF analysis for training episode parquet files."""

from pathlib import Path
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from tqdm import tqdm

INPUT_FOLDER = "./data/rl_training/training_data"
OUTPUT_FOLDER = "results/Correlation Analysis"
FEATURES = ["z_score", "hurst", "position", "signal", "t_pos_norm"]
REQUIRED_COLUMNS = ["z_score", "hurst", "position", "signal", "window"]
MIN_ROWS = 50
FEATURE_LABELS = {
    "z_score": "Z-Score",
    "hurst": "Hurst",
    "position": "Position",
    "signal": "Signal",
    "t_pos_norm": "T\\_Pos Norm",
}


def _fmt_float(value: float, digits: int = 4) -> str:
    if pd.isna(value):
        return "-"
    return f"{float(value):.{digits}f}"


def _build_correlation_latex(pooled_corr: pd.DataFrame, n_episodes: int) -> str:
    ordered_corr = pooled_corr.reindex(index=FEATURES, columns=FEATURES)
    header_cols = " & ".join(FEATURE_LABELS[col] for col in FEATURES)

    rows = []
    for row_feature in FEATURES:
        row_name = FEATURE_LABELS[row_feature]
        values = " & ".join(
            _fmt_float(ordered_corr.loc[row_feature, col]) for col in FEATURES
        )
        rows.append(f"{row_name} & {values} \\\\")

    rows_tex = "\n".join(rows)
    total_x_cols = len(FEATURES)

    return f"""\\begin{{table}}[H]
    \\centering
    \\footnotesize
    \\renewcommand{{\\arraystretch}}{{1.2}}
    \\caption{{Global Interaction Analysis: Pooled Correlation Matrix of State Variables.}}
    \\label{{tab:training-correlation-matrix}}
    \\begin{{tabularx}}{{\\linewidth}}{{l*{{{total_x_cols}}}{{>{{\\centering\\arraybackslash}}X}}}}
    \\toprule
    Metric & {header_cols} \\\\
    \\midrule
    {rows_tex}
    \\bottomrule
    \\end{{tabularx}}
    \\vspace{{1ex}}
    \\raggedright
    \\textit{{Note:}} 
    \\end{{table}}
    """


def _build_vif_latex(pooled_vif: pd.DataFrame) -> str:
    ordered_vif = pooled_vif.reindex(FEATURES)
    header_cols = " & ".join(FEATURE_LABELS[col] for col in FEATURES)
    values = " & ".join(_fmt_float(ordered_vif.loc[col, "VIF"]) for col in FEATURES)
    total_x_cols = len(FEATURES)

    return f"""\\begin{{table}}[H]
    \\centering
    \\footnotesize
    \\renewcommand{{\\arraystretch}}{{1.2}}
    \\caption{{Multicollinearity Diagnostics: Pooled Variance Inflation Factor (VIF) Assessment.}}
    \\label{{tab:training-vif}}
    \\begin{{tabularx}}{{\\linewidth}}{{l*{{{total_x_cols}}}{{>{{\\centering\\arraybackslash}}X}}}}
    \\toprule
    Metric & {header_cols} \\\\
    \\midrule
    VIF & {values} \\\\
    \\bottomrule
    \\end{{tabularx}}
    \\vspace{{1ex}}
    \\raggedright
    \\textit{{Note:}} 
    \\end{{table}}
    """


def _safe_write_csv(df: pd.DataFrame, output_path: Path, filename: str) -> Path:
    target_path = output_path / filename
    try:
        df.to_csv(target_path)
        return target_path
    except PermissionError:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fallback_path = output_path / f"{target_path.stem}_{ts}{target_path.suffix}"
        df.to_csv(fallback_path)
        print(f"Warning: {target_path.name} is locked, saved as {fallback_path.name}")
        return fallback_path


def calculate_t_pos_norm(df: pd.DataFrame) -> pd.Series:
    position = pd.to_numeric(df["position"], errors="coerce").fillna(0.0).to_numpy()
    window = pd.to_numeric(df["window"], errors="coerce").to_numpy()

    t_pos = np.zeros(len(df), dtype=float)
    run_len = 0
    prev_pos = 0.0

    for i, pos in enumerate(position):
        if np.isclose(pos, 0.0):
            run_len = 0
        elif i == 0 or np.isclose(pos, prev_pos):
            run_len = run_len + 1 if run_len > 0 else 1
        else:
            run_len = 1

        t_pos[i] = run_len
        prev_pos = pos

    with np.errstate(divide="ignore", invalid="ignore"):
        t_pos_norm = np.divide(
            t_pos,
            window,
            out=np.full(len(df), np.nan, dtype=float),
            where=window > 0,
        )

    return pd.Series(t_pos_norm, index=df.index, name="t_pos_norm")


def run_post_hoc_analysis() -> None:
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent

    input_path = project_root / INPUT_FOLDER
    output_path = project_root / OUTPUT_FOLDER
    output_path.mkdir(parents=True, exist_ok=True)

    parquet_files = sorted(input_path.glob("*.parquet"))
    print(f"Mechanistic analysis for {len(parquet_files)} files...")

    all_data = []
    skipped_missing_cols = 0
    skipped_low_rows = 0
    skipped_constant = 0
    skipped_errors = 0

    for file_path in tqdm(parquet_files):
        try:
            df = pd.read_parquet(file_path)
            if not all(col in df.columns for col in REQUIRED_COLUMNS):
                skipped_missing_cols += 1
                continue

            df["t_pos_norm"] = calculate_t_pos_norm(df)

            data = (
                df[FEATURES]
                .replace([np.inf, -np.inf], np.nan)
                .dropna()
                .astype(float)
                .copy()
            )

            if len(data) < MIN_ROWS:
                skipped_low_rows += 1
                continue

            if (data.nunique(dropna=False) <= 1).any():
                skipped_constant += 1
                continue

            all_data.append(data)

        except Exception as e:
            skipped_errors += 1
            print(e)
            continue

    if not all_data:
        print("Error: no valid episodes after filtering input data.")
        return

    # Data Pooling
    pooled_df = pd.concat(all_data, ignore_index=True)
    n_episodes = len(all_data)

    # Global Correlation
    pooled_corr = pooled_df.corr()
    corr_csv_path = _safe_write_csv(
        pooled_corr, output_path, "pooled_correlation_matrix.csv"
    )

    corr_latex = _build_correlation_latex(
        pooled_corr=pooled_corr, n_episodes=n_episodes
    )
    with open(
        output_path / "correlation_matrix_table.tex", "w", encoding="utf-8"
    ) as f_tex:
        f_tex.write(corr_latex)

    # Global VIF
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=RuntimeWarning,
            message="divide by zero encountered in scalar divide",
        )
        vif_values = [
            variance_inflation_factor(pooled_df.values, i) for i in range(len(FEATURES))
        ]

    pooled_vif = pd.DataFrame({"VIF": vif_values}, index=FEATURES)
    vif_csv_path = _safe_write_csv(pooled_vif, output_path, "pooled_vif_report.csv")

    vif_latex = _build_vif_latex(pooled_vif=pooled_vif)
    with open(output_path / "vif_table.tex", "w", encoding="utf-8") as f_tex:
        f_tex.write(vif_latex)

    print(
        f"\nSuccess! Processed episodes: {n_episodes} (Pooled rows: {len(pooled_df)})"
    )
    print(f"Saved: {corr_csv_path.name}")
    print(f"Saved: {vif_csv_path.name}")
    print(
        "Skipped -> "
        f"missing_cols: {skipped_missing_cols}, "
        f"low_rows: {skipped_low_rows}, "
        f"constant_or_nan: {skipped_constant}, "
        f"errors: {skipped_errors}"
    )


if __name__ == "__main__":
    run_post_hoc_analysis()
