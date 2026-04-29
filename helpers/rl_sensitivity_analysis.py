"""Script to generate RL Sensitivity Analysis performance reports, including PDF equity plots and formatted LaTeX tables."""

import os
import time
import warnings
import yaml
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

from modules.core.enums import Interval
from modules.performance.stats import calculate_stats
from modules.utils.logger import get_logger

warnings.filterwarnings("ignore", category=UserWarning, module="choreographer")
logger = get_logger(__name__)

FOLDER = "RL Sensitivity Analysis 10x/Wide"

BASELINE = {
    "OOS": "RL OOS 10x",
}

LEVERAGE = 10

ELSEVIER_FONT = "Arial, sans-serif"
FONT_SIZE_TICK = 10
FONT_SIZE_LABEL = 12
FONT_SIZE_TITLE = 13
COLOR_BLACK = "black"

PUBLICATION_COLORS = [
    "#1f77b4",
    "#d62728",
    "#2ca02c",
    "#9467bd",
    "#ff7f0e",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]

PDF_WIDTH = 720
PDF_HEIGHT = 350

SENSITIVITY_PARAMS = [
    "entry_threshold",
    "exit_threshold",
    "stop_loss",
    "top_n",
    "z_score_window",
]

ASSUMPTIONS = [
    "fee_rate",
    "beta_hedge",
    "sl_lock",
    "time_decay_sl",
    "autonomous_agent",
]

NAME_MAP = {
    "entry_threshold": "Entry Threshold",
    "exit_threshold": "Exit Threshold",
    "stop_loss": "Stop Loss",
    "top_n": "Pairs",
    "z_score_window": "Z-Score Window",
    "fee_rate": "Fee Rate",
    "beta_hedge": "Beta Hedge",
    "sl_lock": "SL Lock",
    "time_decay_sl": "Time Decay SL",
    "autonomous_agent": "Risk Management Overlay",
}


def generate_reports(folder_name: str, baseline_dict: dict):
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    category_dir = project_root / "results" / folder_name

    oos_base_dir = category_dir / "oos"

    report_output_dir = category_dir / "sensitivity_report"
    report_output_dir.mkdir(parents=True, exist_ok=True)

    if not oos_base_dir.exists():
        raise ValueError(f"Directory 'oos' must exist inside {category_dir}")

    axis_style_x = dict(
        showline=True,
        linewidth=1,
        linecolor=COLOR_BLACK,
        mirror=True,
        ticks="inside",
        tickcolor=COLOR_BLACK,
        tickwidth=1,
        tickangle=0,
        title_standoff=5,
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
        title_standoff=5,
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
        y=-0.10,
        xanchor="center",
        x=0.5,
        font=dict(family=ELSEVIER_FONT, size=FONT_SIZE_TICK, color=COLOR_BLACK),
        bgcolor="rgba(255, 255, 255, 0)",
        bordercolor=COLOR_BLACK,
        borderwidth=0,
    )

    def get_run_data(run_dir: Path):
        config_path = run_dir / ".hydra" / "config.yaml"
        ts_files = list(run_dir.glob("returns_multi_pair_*.parquet"))
        exec_files = list(run_dir.glob("exec_logger_*.parquet"))

        missing = []
        if not config_path.exists():
            missing.append("config.yaml")
        if not ts_files:
            missing.append("returns_multi_pair_*.parquet")
        if not exec_files:
            missing.append("exec_logger_*.parquet")

        if missing:
            logger.warning(
                f"[{run_dir.name}] REJECTED: Files not found -> {', '.join(missing)}"
            )
            return None

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)

            ts_df = pd.read_parquet(ts_files[0])
            exec_df = pd.read_parquet(exec_files[0])

            def get_cfg(key):
                if key in config:
                    return config[key]
                for section in ["performance", "pair_selection", "market", "settings"]:
                    if section in config and key in config[section]:
                        return config[section][key]
                return None

            params = {k: get_cfg(k) for k in (SENSITIVITY_PARAMS + ASSUMPTIONS)}

            initial_cash = ts_df["equity"].iloc[0]
            ret_series = (ts_df["equity"] / initial_cash) - 1

            stats_files = list(run_dir.glob("stats_*.parquet"))
            if stats_files:
                df_stats = pd.read_parquet(stats_files[0])
                if "metric" in df_stats.columns:
                    df_stats = df_stats.set_index("metric")
                net_stats = df_stats["net"]
            else:
                risk_free_rate = float(
                    config.get("market", {}).get("risk_free_rate_annual", 0.0)
                )
                stats_calc = calculate_stats(
                    ts_df, exec_df, initial_cash, Interval.H1, risk_free_rate
                )
                net_stats = stats_calc["net"]

            return {
                "params": params,
                "metrics": net_stats,
                "ret_series": ret_series,
                "id": run_dir.name,
            }
        except Exception as e:
            logger.error(
                f"[{run_dir.name}] ERROR DURING LOADING: {type(e).__name__} - {str(e)}"
            )
            return None

    logger.info("Loading baseline model...")
    base_oos = get_run_data(oos_base_dir / baseline_dict["OOS"])

    if not base_oos:
        raise ValueError("Error loading Baseline OOS.")

    base_oos_series = base_oos["ret_series"] - base_oos["ret_series"].iloc[0]

    sensitivity_dict = {p: {} for p in SENSITIVITY_PARAMS}
    mechanism_dict = {p: {} for p in ASSUMPTIONS}

    def is_different(val1, val2):
        s1, s2 = str(val1).lower().strip(), str(val2).lower().strip()
        if s1 == s2:
            return False
        try:
            return float(val1) != float(val2)
        except (ValueError, TypeError):
            return True

    logger.info("Extracting and sorting OOS runs...")
    for oos_run_dir in oos_base_dir.iterdir():
        if not oos_run_dir.is_dir() or oos_run_dir.name == baseline_dict["OOS"]:
            continue

        run_oos = get_run_data(oos_run_dir)
        if not run_oos:
            continue

        diffs_sens = [
            k
            for k in SENSITIVITY_PARAMS
            if is_different(run_oos["params"][k], base_oos["params"][k])
        ]
        diffs_assump = [
            k
            for k in ASSUMPTIONS
            if is_different(run_oos["params"][k], base_oos["params"][k])
        ]

        added = False

        if len(diffs_sens) == 1 and len(diffs_assump) == 0:
            p_name = diffs_sens[0]
            val = run_oos["params"][p_name]

            run_oos["series_oos"] = (
                run_oos["ret_series"] - run_oos["ret_series"].iloc[0]
            )

            sensitivity_dict[p_name][val] = run_oos
            added = True
            logger.info(f"[{oos_run_dir.name}] ADDED (Sensitivity): {p_name} = {val}")

        elif len(diffs_assump) == 1 and len(diffs_sens) == 0:
            p_name = diffs_assump[0]
            val = run_oos["params"][p_name]

            if p_name == "beta_hedge" and val == "no_hedge":
                val = False
            elif p_name == "fee_rate":
                val = f"{float(val):.2%}"
            elif p_name == "autonomous_agent":
                if isinstance(val, bool):
                    val = not val
                elif isinstance(val, str):
                    if val.lower() == "true":
                        val = False
                    elif val.lower() == "false":
                        val = True

            run_oos["series_oos"] = (
                run_oos["ret_series"] - run_oos["ret_series"].iloc[0]
            )

            mechanism_dict[p_name][val] = run_oos
            added = True
            logger.info(f"[{oos_run_dir.name}] ADDED (Mechanism): {p_name} = {val}")

        if not added:
            logger.warning(
                f"[{oos_run_dir.name}] SKIPPED: Diffs count not match. Diffs SENS: {diffs_sens}, Diffs ASSUMP: {diffs_assump}"
            )

    def plot_and_save(group_dict, prefix):
        for param_name, variations in group_dict.items():
            if not variations:
                continue

            fig = make_subplots(
                rows=1,
                cols=1,
                subplot_titles=[
                    "Out-of-Sample Performance (2025)",
                ],
            )

            fig.add_trace(
                go.Scatter(
                    x=base_oos_series.index,
                    y=base_oos_series,
                    name="Agent 2",
                    legendgroup="Agent 2",
                    line=dict(color="#FF8C00", width=1.5),
                    showlegend=True,
                ),
            )

            sorted_vals = sorted(
                variations.keys(),
                key=lambda x: (
                    (0, float(x))
                    if str(x).lstrip("-").replace(".", "", 1).isdigit()
                    else (1, str(x))
                ),
            )

            for i, val in enumerate(sorted_vals):
                color = PUBLICATION_COLORS[i % len(PUBLICATION_COLORS)]
                v_data = variations[val]
                var_label = f"{NAME_MAP.get(param_name, param_name)} = {val}"

                fig.add_trace(
                    go.Scatter(
                        x=v_data["series_oos"].index,
                        y=v_data["series_oos"],
                        name=var_label,
                        legendgroup=var_label,
                        line=dict(color=color, width=1.5),
                        showlegend=True,
                    ),
                )

            fig.update_layout(
                width=PDF_WIDTH,
                height=PDF_HEIGHT,
                font=dict(
                    family=ELSEVIER_FONT, size=FONT_SIZE_LABEL, color=COLOR_BLACK
                ),
                plot_bgcolor="white",
                paper_bgcolor="white",
                showlegend=True,
                legend=legend_style,
                margin=dict(t=30, b=40, l=45, r=10),
            )

            for annotation in fig["layout"]["annotations"]:
                annotation["font"] = dict(
                    family=ELSEVIER_FONT, size=FONT_SIZE_TITLE, color=COLOR_BLACK
                )
                annotation["yshift"] = 5

            fig.update_xaxes(
                **axis_style_x,
                tickformat="%b\n%Y",
                dtick="M3",
                tick0=base_oos_series.index[0] if len(base_oos_series) > 0 else None,
            )

            fig.update_yaxes(**axis_style_y, tickformat=".0%")
            fig.update_yaxes(title_text="Cumulative Return")

            pdf_path = (
                report_output_dir
                / f"{prefix}_{NAME_MAP.get(param_name, param_name).replace(' ', '_')}.pdf"
            )
            try:
                fig.write_image(str(pdf_path), format="pdf")
                logger.info(f"PDF saved: {pdf_path.name}")
            except Exception as e:
                logger.warning(
                    f"Initial save failed for {pdf_path.name}. Killing kaleido and retrying... {e}"
                )
                os.system("taskkill /F /IM kaleido.exe /T >nul 2>&1")
                time.sleep(1)

                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        fig.write_image(str(pdf_path), format="pdf")
                        logger.info(
                            f"PDF saved on retry {attempt + 1}: {pdf_path.name}"
                        )
                        break
                    except Exception as retry_e:
                        if attempt < max_retries - 1:
                            logger.warning(
                                f"Retry {attempt + 1} failed for {pdf_path.name}. Killing kaleido again..."
                            )
                            os.system("taskkill /F /IM kaleido.exe /T >nul 2>&1")
                            time.sleep(2)
                        else:
                            logger.error(
                                f"Failed to save {pdf_path.name} after {max_retries} retries: {retry_e}"
                            )

    def generate_latex_table(results_dict, params_list, prefix, title, label):
        active_params = [p for p in params_list if results_dict.get(p)]
        if not active_params:
            return

        num_vars = sum(len(results_dict[p]) for p in active_params)
        total_cols = 2 + num_vars
        col_format = "l" + f"*{{{total_cols - 1}}}{{>{{\\centering\\arraybackslash}}X}}"

        header1 = "& Agent 2"
        header2 = "& -"

        for p in active_params:
            vals = sorted(
                results_dict[p].keys(),
                key=lambda x: (
                    (0, float(x))
                    if str(x).lstrip("-").replace(".", "", 1).isdigit()
                    else (1, str(x))
                ),
            )
            header1 += f" & \\multicolumn{{{len(vals)}}}{{c}}{{{NAME_MAP.get(p, p)}}}"
            for v in vals:
                v_str = f"{v:g}" if isinstance(v, float) else str(v)
                header2 += f" & {v_str}"

        row_groups = [
            [
                ("cagr", "CAGR", "%"),
                ("volatility_annual", "Annual Volatility", "%"),
                ("max_drawdown", "Max Drawdown", "%"),
            ],
            [
                ("win_count", "Win Count", "d"),
                ("lose_count", "Loss Count", "d"),
                ("win_rate", "Win Rate", "%"),
            ],
            [
                ("avg_win_return", "Avg Win Return", "%"),
                ("avg_lose_return", "Avg Loss Return", "%"),
                ("avg_trade_return", "Avg Trade Return", "%"),
                ("avg_trade_duration", "Avg Trade Duration", "f"),
            ],
            [
                ("sharpe_ratio_annual", "Sharpe Ratio (Ann.)", "f4"),
                ("sortino_ratio_annual", "Sortino Ratio (Ann.)", "f4"),
                ("calmar_ratio", "Calmar Ratio", "f4"),
            ],
        ]

        def format_val(val, fmt):
            if pd.isnull(val):
                return "-"
            if fmt == "%":
                return f"{val * 100:.2f}\\%"
            if fmt == "d":
                return f"{int(val)}"
            if fmt == "f":
                return f"{val:.2f}"
            if fmt == "f4":
                return f"{val:.4f}"
            return str(val)

        rows_tex = ""
        for group in row_groups:
            for orig_name, tex_name, fmt in group:
                row_str = f"{tex_name} & {format_val(base_oos['metrics'].get(orig_name), fmt)}"
                for p in active_params:
                    vals = sorted(
                        results_dict[p].keys(),
                        key=lambda x: (
                            (0, float(x))
                            if str(x).lstrip("-").replace(".", "", 1).isdigit()
                            else (1, str(x))
                        ),
                    )
                    for v in vals:
                        metric_val = results_dict[p][v]["metrics"].get(orig_name)
                        row_str += f" & {format_val(metric_val, fmt)}"

                row_str += " \\\\"
                if orig_name in ["max_drawdown", "win_rate", "avg_trade_duration"]:
                    row_str += "[4pt]"
                rows_tex += row_str + "\n"

        fee = float(base_oos["params"].get("fee_rate", 0.0))
        baseline_str = ", ".join(
            [
                f"{NAME_MAP.get(p, p)} = {base_oos['params'].get(p, 'N/A')}"
                for p in params_list
            ]
        )
        note = f"Agent 2 ({fee * 100:.2f}\\% fees, leverage {int(LEVERAGE)}x): {baseline_str}."

        tex = f"""\\begin{{landscape}}
\\vspace*{{\\fill}}
\\renewcommand{{\\arraystretch}}{{1.2}}
\\begin{{center}}
\\footnotesize
\\captionof{{table}}{{{title}}}
\\vspace{{12pt}}
\\label{{{label}}}
\\begin{{tabularx}}{{\\linewidth}}{{{col_format}}}
\\toprule
 {header1} \\\\
 {header2} \\\\
\\midrule
{rows_tex.strip()}
\\bottomrule
\\end{{tabularx}}

\\vspace{{12pt}}
\\justifying \\noindent \\scriptsize Note: {note}
\\end{{center}}
\\vspace*{{\\fill}}
\\end{{landscape}}"""

        with open(
            report_output_dir / f"{prefix}_table.tex", "w", encoding="utf-8"
        ) as f:
            f.write(tex)
        logger.info(f"LaTeX Table saved: {prefix}_table.tex")

    def generate_mechanism_latex_table(results_dict, params_list, prefix, title, label):
        active_params = [p for p in params_list if results_dict.get(p)]
        if not active_params:
            return

        num_vars = sum(len(results_dict[p]) for p in active_params)
        total_cols = 2 + num_vars
        col_format = "l" + f"*{{{total_cols - 1}}}{{>{{\\centering\\arraybackslash}}X}}"

        header1 = "& Agent 2"
        header2 = "& -"

        for p in active_params:
            vals = sorted(
                results_dict[p].keys(),
                key=lambda x: (
                    (0, float(x))
                    if str(x).lstrip("-").replace(".", "", 1).isdigit()
                    else (1, str(x))
                ),
            )
            header1 += f" & \\multicolumn{{{len(vals)}}}{{c}}{{{NAME_MAP.get(p, p)}}}"
            for v in vals:
                v_str = f"{v:g}" if isinstance(v, float) else str(v)
                v_str = v_str.replace("%", "\\%")
                header2 += f" & {v_str}"

        row_groups = [
            [
                ("cagr", "CAGR", "%"),
                ("volatility_annual", "Annual Volatility", "%"),
                ("max_drawdown", "Max Drawdown", "%"),
            ],
            [
                ("win_count", "Win Count", "d"),
                ("lose_count", "Loss Count", "d"),
                ("win_rate", "Win Rate", "%"),
            ],
            [
                ("avg_win_return", "Avg Win Return", "%"),
                ("avg_lose_return", "Avg Loss Return", "%"),
                ("avg_trade_return", "Avg Trade Return", "%"),
                ("avg_trade_duration", "Avg Trade Duration", "f"),
            ],
            [
                ("sharpe_ratio_annual", "Sharpe Ratio (Ann.)", "f4"),
                ("sortino_ratio_annual", "Sortino Ratio (Ann.)", "f4"),
                ("calmar_ratio", "Calmar Ratio", "f4"),
            ],
        ]

        def format_val(val, fmt):
            if pd.isnull(val):
                return "-"
            if fmt == "%":
                return f"{val * 100:.2f}\\%"
            if fmt == "d":
                return f"{int(val)}"
            if fmt == "f":
                return f"{val:.2f}"
            if fmt == "f4":
                return f"{val:.4f}"
            return str(val)

        rows_tex = ""
        for group in row_groups:
            for orig_name, tex_name, fmt in group:
                row_str = f"{tex_name} & {format_val(base_oos['metrics'].get(orig_name), fmt)}"
                for p in active_params:
                    vals = sorted(
                        results_dict[p].keys(),
                        key=lambda x: (
                            (0, float(x))
                            if str(x).lstrip("-").replace(".", "", 1).isdigit()
                            else (1, str(x))
                        ),
                    )
                    for v in vals:
                        metric_val = results_dict[p][v]["metrics"].get(orig_name)
                        row_str += f" & {format_val(metric_val, fmt)}"

                row_str += " \\\\"
                if orig_name in ["max_drawdown", "win_rate", "avg_trade_duration"]:
                    row_str += "[4pt]"
                rows_tex += row_str + "\n"

        fee = float(base_oos["params"].get("fee_rate", 0.0))
        baseline_params = []
        for p in params_list:
            if p != "fee_rate":
                val = base_oos["params"].get(p, "N/A")
                baseline_params.append(f"{NAME_MAP.get(p, p)} = {val}")

        baseline_str = ", ".join(baseline_params)
        note = f"Agent 2 ({fee * 100:.2f}\\% fees, leverage {int(LEVERAGE)}x): {baseline_str}."

        tex = f"""\\begin{{landscape}}
\\vspace*{{\\fill}}
\\renewcommand{{\\arraystretch}}{{1.2}}
\\begin{{center}}
\\footnotesize
\\captionof{{table}}{{{title}}}
\\vspace{{12pt}}
\\label{{{label}}}
\\begin{{tabularx}}{{\\linewidth}}{{{col_format}}}
\\toprule
 {header1} \\\\
 {header2} \\\\
\\midrule
{rows_tex.strip()}
\\bottomrule
\\end{{tabularx}}

\\vspace{{12pt}}
\\justifying \\noindent \\scriptsize Note: {note}
\\end{{center}}
\\vspace*{{\\fill}}
\\end{{landscape}}"""

        with open(
            report_output_dir / f"{prefix}_table.tex", "w", encoding="utf-8"
        ) as f:
            f.write(tex)
        logger.info(f"LaTeX Table saved: {prefix}_table.tex")

    logger.info("Generating PDF Plots...")
    plot_and_save(sensitivity_dict, "rl_sensitivity")
    plot_and_save(mechanism_dict, "rl_mechanism")

    logger.info("Generating LaTeX Tables...")
    generate_latex_table(
        sensitivity_dict,
        SENSITIVITY_PARAMS,
        "sensitivity",
        "Sensitivity Analysis of Out-Of-Sample Agent Performance (2025).",
        "tab:rl_oos_sensitivity",
    )
    generate_mechanism_latex_table(
        mechanism_dict,
        ASSUMPTIONS,
        "mechanism",
        "Assumptions Verification: Out-Of-Sample Agent Performance (2025).",
        "tab:rl_oos_mechanism",
    )

    logger.info("RL Sensitivity Pipeline completed successfully.")

    logger.info("Ending...")
    os._exit(0)


if __name__ == "__main__":
    generate_reports(FOLDER, BASELINE)
