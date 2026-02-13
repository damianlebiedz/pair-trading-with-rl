import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Literal
import pandas as pd
from omegaconf import DictConfig

from modules.performance.models import StrategyResult
from modules.core.statistical_tests import (
    johansen_cointegration,
    engle_granger_cointegration,
)
from modules.data_services.data_loaders import load_data
from modules.data_services.data_utils import (
    save_strategy_result,
    load_btc_benchmark,
    save_dataframe,
    load_strategy_result,
)
from modules.performance.multi_pair_optimizer import MultiPairOptimizer
from modules.data_services.merge_utils import (
    aggregate_strategy_results,
    stitch_strategy_results,
)
from modules.performance.objectives import SortinoWithPenalty
from modules.performance.stats import calculate_stats
from modules.performance.strategy import Strategy
from modules.utils.plots import plot_returns, plot_zscore_pos, plot_spread_pos

logger = logging.getLogger(__name__)


def setup_run_environment(calling_file: str) -> str:
    script_dir = os.path.dirname(os.path.abspath(calling_file))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = os.path.join(project_root, "results", timestamp)

    os.makedirs(output_dir, exist_ok=True)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    file_handler = logging.FileHandler(os.path.join(output_dir, "execution.log"))
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    if not any(isinstance(h, logging.StreamHandler) for h in root_logger.handlers):
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    logger.debug("--- Environment Setup ---")
    logger.debug(f"Output Directory: {output_dir}")

    return output_dir


def setup_rl_run_environment(calling_file: str) -> str:
    script_dir = os.path.dirname(os.path.abspath(calling_file))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = os.path.join(project_root, "data_rl")

    os.makedirs(output_dir, exist_ok=True)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    file_handler = logging.FileHandler(os.path.join(output_dir, f"{timestamp}.log"))
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    if not any(isinstance(h, logging.StreamHandler) for h in root_logger.handlers):
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    logger.debug("--- RL Environment Setup ---")
    logger.debug(f"Output Directory: {output_dir}")

    return output_dir


def execute_optimization(
    cfg: DictConfig,
    bt: Strategy,
    static_params: dict[str, Any],
    param_space: list[Any],
    metric_type: Literal["gross", "net"],
    objective_func: Literal["sortino"],
) -> dict[str, Any]:

    win_opt_start = cfg.performance.optimization.win_start
    opt_start = cfg.performance.optimization.start
    opt_end = cfg.performance.optimization.end

    if objective_func not in ["sortino"]:
        raise ValueError("objective_func should be 'sortino'")

    if objective_func == "sortino":
        objective_func = SortinoWithPenalty(
            min_trades_per_pair=cfg.performance.optimization.min_trades_per_pair
        )

    best_params, best_score = bt.run_optimization(
        static_params=static_params,
        param_space=param_space,
        metric_type=metric_type,
        objective_func=objective_func,
        opt_start=opt_start,
        opt_end=opt_end,
        win_opt_start=win_opt_start,
        n_iter=cfg.performance.optimization.n_iter,
        replicates=cfg.performance.optimization.replicates,
        penalty_bad=cfg.performance.optimization.penalty_bad,
    )

    best_params.update(static_params)
    log = (best_params, best_score)
    logger.info(log)

    return best_params


def execute_multi_pair_optimization(
    cfg: DictConfig,
    strategies: list[Strategy],
    static_params: dict[str, Any],
    param_space: list[Any],
    metric_type: Literal["gross", "net"],
    objective_func: Literal["sortino"],
) -> dict[str, Any]:
    logger.info(f"Starting Multi-Pair Optimization on {len(strategies)} pairs...")

    if objective_func not in ["sortino"]:
        raise ValueError("objective_func should be 'sortino'")

    if objective_func == "sortino":
        objective_func = SortinoWithPenalty(
            min_trades_per_pair=cfg.performance.optimization.min_trades_per_pair
        )

    optimizer = MultiPairOptimizer(strategies, cfg)

    best_params, best_score = optimizer.run(
        static_params=static_params,
        param_space=param_space,
        metric_type=metric_type,
        objective_func=objective_func,
    )

    best_params.update(static_params)
    log = (best_params, best_score)
    logger.info(log)

    return best_params


def execute_testing(
    cfg: DictConfig,
    bt: Strategy,
    best_params: dict[str, Any],
    ticker_x: str,
    ticker_y: str,
    output_dir: str,
    win_test_start: str,
    test_start: str,
    test_end: str,
    subdir: str | None = None,
) -> StrategyResult:

    if subdir:
        output_dir = os.path.join(output_dir, subdir)

    window_factor = best_params["window_factor"]
    entry_threshold = best_params["entry_threshold"]
    exit_threshold = best_params["exit_threshold"]
    stop_loss = best_params["stop_loss"]

    result = bt.run_strategy(
        window_factor=window_factor,
        entry_threshold=entry_threshold,
        exit_threshold=exit_threshold,
        stop_loss=stop_loss,
        test_start=test_start,
        test_end=test_end,
        win_test_start=win_test_start,
    )

    save_strategy_result(
        result=result,
        file_name=f"returns_{ticker_x}_{ticker_y}_{test_start}_{test_end}",
        directory=output_dir,
    )

    save_dataframe(
        df=result.exec_logger,
        file_name=f"exec_logger_{ticker_x}_{ticker_y}_{test_start}_{test_end}",
        directory=output_dir,
    )

    save_dataframe(
        df=result.stats,
        file_name=f"stats_{ticker_x}_{ticker_y}_{test_start}_{test_end}",
        directory=output_dir,
    )

    plot_zscore_pos(result, directory=output_dir, save=True)
    plot_spread_pos(result, directory=output_dir, save=True)
    btc_data = load_btc_benchmark(
        test_start=test_start,
        test_end=test_end,
        interval=cfg.market.interval,
    )
    plot_returns(result, btc_data, directory=output_dir, save=True)

    logger.info("Testing completed, returning StrategyResult.")

    return result


def execute_pair_selection(
    tickers: list[str],
    test_start: str,
    test_end: str,
    interval: str,
    method: str,
    ps_factor: float,
    top_n_factor: float,
    output_dir: str,
    coint_type: Literal["eg", "johansen"],
    opt_start: str | None = None,
    opt_end: str | None = None,
) -> pd.DataFrame:
    selection_method = method
    if selection_method not in ["both", "second"]:
        raise ValueError("'method' must be 'both' or 'second'")

    if coint_type not in ["eg", "johansen"]:
        raise ValueError(f"Unknown cointegration type: {coint_type}")

    logger.info(f"Running pair selection using '{coint_type}' cointegration test.")

    df_test = load_data(
        tickers=tickers,
        start=test_start,
        end=test_end,
        interval=interval,
    )

    if coint_type == "johansen":
        ps_df_test = johansen_cointegration(df_test)
        if ps_factor == 0.05:
            crit_col = "crit_95"
        elif ps_factor == 0.01:
            crit_col = "crit_99"
        else:
            raise ValueError("ps_factor should be 0.05 or 0.01 for Johansen")

        factor = ps_df_test[crit_col]

    else:
        ps_df_test = engle_granger_cointegration(df_test)
        factor = ps_factor
        crit_col = None

    if selection_method == "second" or opt_start is None or opt_end is None:
        logger.info("Selection method: 'second' (filtering by test set only).")
        start = test_start

        if coint_type == "johansen":
            condition = (ps_df_test["max_eig_stat"] > factor) & (ps_df_test["beta"] > 0)
            sort_column = "max_eig_stat"
            ascending = False
        else:
            condition = (ps_df_test["p_value"] < factor) & (ps_df_test["beta"] > 0)
            sort_column = "p_value"
            ascending = True

        final_df_source = ps_df_test[condition]

    else:
        logger.info(
            "Selection method: 'both' (filtering by optimization AND test sets)."
        )
        start = opt_start

        df_opt = load_data(
            tickers=tickers,
            start=opt_start,
            end=opt_end,
            interval=interval,
        )

        if coint_type == "johansen":
            ps_df_opt = johansen_cointegration(df_opt)

            merged_df = pd.merge(
                ps_df_opt,
                ps_df_test,
                on="pair",
                how="inner",
                suffixes=("_opt", "_test"),
            )

            if merged_df.empty:
                logger.warning(
                    "Intersection of Opt and Test sets is empty. No pairs to analyze."
                )
                return pd.DataFrame()

            crit_col_test = f"{crit_col}_test"
            crit_col_opt = f"{crit_col}_opt"

            condition = (
                (merged_df["max_eig_stat_opt"] > merged_df[crit_col_opt])
                & (merged_df["max_eig_stat_test"] > merged_df[crit_col_test])
                & (merged_df["beta_opt"] > 0)
                & (merged_df["beta_test"] > 0)
            )
            sort_column = "max_eig_stat_test"
            ascending = False

        else:
            ps_df_opt = engle_granger_cointegration(df_opt)

            merged_df = pd.merge(
                ps_df_opt,
                ps_df_test,
                on="pair",
                how="inner",
                suffixes=("_opt", "_test"),
            )

            if merged_df.empty:
                logger.warning(
                    "Intersection of Opt and Test sets is empty. No pairs to analyze."
                )
                return pd.DataFrame()

            condition = (
                (merged_df["p_value_opt"] < factor)
                & (merged_df["p_value_test"] < factor)
                & (merged_df["beta_opt"] > 0)
                & (merged_df["beta_test"] > 0)
            )
            sort_column = "p_value_test"
            ascending = True

        final_df_source = merged_df[condition]

        if not final_df_source.empty:
            final_df_source = final_df_source.copy()
            final_df_source["beta"] = final_df_source["beta_test"]

    if final_df_source.empty:
        logger.warning(
            f"No pairs met the statistical significance criteria (method='{method}', type='{coint_type}')."
        )
        return pd.DataFrame()

    final_df = (
        final_df_source.sort_values(sort_column, ascending=ascending)
        .head(top_n_factor)
        .reset_index(drop=True)
        .round(4)
    )

    if final_df.empty:
        logger.warning(
            "Pairs met significance criteria but were filtered out due to negative beta."
        )
        return pd.DataFrame()

    save_dataframe(
        df=final_df,
        file_name=f"pair_selection_{method}_{coint_type}_{start}_{test_end}",
        directory=output_dir,
    )

    logger.info(f"Pair Selection completed. Selected {len(final_df)} pairs.")

    return final_df


def merge_multi_pair_results(
    cfg: DictConfig,
    output_dir: str,
    results: list[StrategyResult],
    initial_cash: float,
    risk_free_rate_annual: float,
    test_start: str,
    test_end: str,
) -> StrategyResult:
    """Merges multiple StrategyResult objects into one aggregate result and saves it."""
    if not results:
        raise ValueError("No results to merge")

    merged_df, merged_exec_res = aggregate_strategy_results(results, initial_cash)

    stats = calculate_stats(
        df=merged_df,
        exec_log_df=merged_exec_res,
        initial_cash=initial_cash,
        interval=results[0].interval,
        risk_free_rate_annual=risk_free_rate_annual,
    )

    final_result = StrategyResult(
        data=merged_df,
        ticker_x="multi",
        ticker_y="pair",
        start=test_start,
        end=test_end,
        interval=results[0].interval,
        fee_rate=results[0].fee_rate,
        stats=stats,
        exec_logger=merged_exec_res,
    )

    save_strategy_result(
        result=final_result,
        file_name=f"returns_multi_pair_{test_start}_{test_end}",
        directory=output_dir,
    )

    save_dataframe(
        df=merged_exec_res,
        file_name=f"exec_logger_multi_pair_{final_result.start}_{final_result.end}",
        directory=output_dir,
    )

    save_dataframe(
        df=final_result.stats,
        file_name=f"stats_multi_pair_{test_start}_{test_end}",
        directory=output_dir,
    )

    btc_data = load_btc_benchmark(
        test_start=final_result.start,
        test_end=final_result.end,
        interval=cfg.market.interval,
    )
    plot_returns(final_result, btc_data, directory=output_dir, save=True)

    logger.info("Merge Multi-Pair Results completed, returning StrategyResult.")

    return final_result


def merge_multi_period_results(
    cfg: DictConfig,
    output_dir: str,
    ticker_x: str,
    ticker_y: str,
    initial_cash: float,
    risk_free_rate_annual: float,
) -> StrategyResult | None:
    """
    Checks for multiple iteration folders (1, 2, ...) and merges results for a specific pair
    chronologically using the stitch_strategy_results helper.
    """
    base_path = Path(output_dir)

    if not (base_path / "1").exists():
        logger.warning(
            f"Cannot perform multi-period merge when there is no multi periods in {output_dir}"
        )
        return None

    logger.info(
        f"Detected multi-period structure in {output_dir}. Merging results for {ticker_x}-{ticker_y}..."
    )

    iter_dirs = sorted(
        [d for d in base_path.iterdir() if d.is_dir() and d.name.isdigit()],
        key=lambda x: int(x.name),
    )

    results = []

    for d in iter_dirs:
        pattern = f"returns_{ticker_x}_{ticker_y}_*.parquet"
        files = list(d.glob(pattern))

        if not files:
            logger.warning(f"No result file found for {ticker_x}-{ticker_y} in {d}")
            continue

        file_stem = files[0].stem
        res = load_strategy_result(file_stem, directory=str(d))
        results.append(res)

    if not results:
        logger.warning("No results collected for merging.")
        return None

    final_df, final_exec_df = stitch_strategy_results(results)

    stats = calculate_stats(
        df=final_df,
        exec_log_df=final_exec_df,
        initial_cash=initial_cash,
        interval=results[0].interval,
        risk_free_rate_annual=risk_free_rate_annual,
    )

    final_result = StrategyResult(
        data=final_df,
        ticker_x=ticker_x,
        ticker_y=ticker_y,
        start=results[0].start,
        end=results[-1].end,
        interval=results[0].interval,
        fee_rate=results[0].fee_rate,
        stats=stats,
        exec_logger=final_exec_df,
    )

    save_strategy_result(
        result=final_result,
        file_name=f"returns_{ticker_x}_{ticker_y}_{final_result.start}_{final_result.end}",
        directory=output_dir,
    )

    save_dataframe(
        df=final_exec_df,
        file_name=f"exec_logger_{ticker_x}_{ticker_y}_{final_result.start}_{final_result.end}",
        directory=output_dir,
    )

    save_dataframe(
        df=final_result.stats,
        file_name=f"stats_{ticker_x}_{ticker_y}_{final_result.start}_{final_result.end}",
        directory=output_dir,
    )

    btc_data = load_btc_benchmark(
        test_start=final_result.start,
        test_end=final_result.end,
        interval=cfg.market.interval,
    )
    plot_returns(final_result, btc_data, directory=output_dir, save=True)

    logger.info(f"Merge Multi-Period Results for {ticker_x}-{ticker_y} completed.")

    return final_result
