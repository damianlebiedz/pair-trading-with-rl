import logging
import os
import sys
from datetime import datetime
from typing import Any
import pandas as pd
from omegaconf import DictConfig

from modules.core.models import StrategyResult
from modules.core.statistical_tests import engle_granger_cointegration
from modules.data_services.data_loaders import load_data
from modules.data_services.data_utils import (
    save_strategy_result,
    load_btc_benchmark,
    save_dataframe,
)
from modules.performance.optimization import MultiPairOptimizer
from modules.performance.stats import (
    calculate_multi_pair_stats,
    aggregate_strategy_results,
)
from modules.performance.strategy import Strategy
from modules.visualization.plots import plot_returns, plot_zscore_pos

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


def execute_optimization(
    cfg: DictConfig,
    bt: Strategy,
    static_params: dict[str, Any],
    param_space: list[Any],
    metric: tuple[str, str],
) -> dict[str, Any]:

    beta_opt_start = cfg.performance.optimization.beta_start
    opt_start = cfg.performance.optimization.start
    opt_end = cfg.performance.optimization.end

    best_params, best_score = bt.run_optimization(
        static_params=static_params,
        param_space=param_space,
        metric=metric,
        opt_start=opt_start,
        opt_end=opt_end,
        beta_opt_start=beta_opt_start,
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
    metric: tuple[str, str],
) -> dict[str, Any]:
    logger.info(f"Starting Multi-Pair Optimization on {len(strategies)} pairs...")

    optimizer = MultiPairOptimizer(strategies, cfg)

    best_params, best_score = optimizer.run(
        static_params=static_params, param_space=param_space, metric=metric
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
    beta_test_start: str,
    test_start: str,
    test_end: str,
) -> StrategyResult:

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
        beta_test_start=beta_test_start,
    )

    save_strategy_result(
        result=result,
        file_name=f"test_{ticker_x}_{ticker_y}_{test_start}_{test_end}",
        directory=output_dir,
    )

    save_dataframe(
        df=result.stats,
        file_name=f"test_stats_{ticker_x}_{ticker_y}_{test_start}_{test_end}",
        directory=output_dir,
    )

    plot_zscore_pos(result, directory=output_dir, save=True, show=True)
    btc_data = load_btc_benchmark(
        test_start=test_start,
        test_end=test_end,
        interval=cfg.market.interval,
    )
    plot_returns(result, btc_data, directory=output_dir, save=True, show=True)

    logger.info("Testing completed, returning StrategyResult.")

    return result


def execute_pair_selection(cfg: DictConfig, output_dir: str) -> pd.DataFrame:
    df_opt = load_data(
        tickers=cfg.tickers,
        start=cfg.pair_selection.optimization.start,
        end=cfg.pair_selection.optimization.end,
        interval=cfg.market.interval,
    )

    df_test = load_data(
        tickers=cfg.tickers,
        start=cfg.pair_selection.test.start,
        end=cfg.pair_selection.test.end,
        interval=cfg.market.interval,
    )

    eg_df_opt = engle_granger_cointegration(df_opt, source="log_prices")
    eg_df_test = engle_granger_cointegration(df_test, source="log_prices")

    eg_factor = cfg.pair_selection.eg_factor
    selection_method = cfg.pair_selection.method
    if selection_method not in ["both", "second"]:
        raise ValueError("'method' must be 'both' or 'second'")

    merged_df = pd.merge(
        eg_df_opt, eg_df_test, on="pair", how="inner", suffixes=("_opt", "_test")
    )

    if selection_method == "second":
        condition = merged_df["eg_p_value_test"] <= eg_factor
        sort_column = "eg_p_value_test"
        logger.info("Selection method: 'second' (filtering by test set only).")

    else:
        condition = (merged_df["eg_p_value_opt"] <= eg_factor) & (
            merged_df["eg_p_value_test"] <= eg_factor
        )
        sort_column = "eg_p_value_opt"
        logger.info(
            "Selection method: 'both' (filtering by optimization AND test sets)."
        )

    final_df = merged_df[condition].sort_values(sort_column).reset_index(drop=True)

    save_dataframe(
        df=final_df,
        file_name=f"pair_selection_{selection_method}_{cfg.pair_selection.optimization.start}_{cfg.pair_selection.test.end}",
        directory=output_dir,
    )

    logger.info(
        f"Pair Selection completed. Selected {len(final_df)} pairs using method '{selection_method}' with eg_factor <= {eg_factor}."
    )

    return final_df


def merge_multi_pair_results(
    cfg: DictConfig,
    output_dir: str,
    results: list[StrategyResult],
    individual_stats_dfs: list[pd.DataFrame],
    total_initial_cash: float,
    risk_free_rate_annual: float,
    test_start: str,
    test_end: str,
) -> StrategyResult:
    """Merges multiple StrategyResult objects into one aggregate result, saves it and shows PnL plot."""
    if not results:
        raise ValueError("No results to merge")

    merged_df = aggregate_strategy_results(results, total_initial_cash)

    stats = calculate_multi_pair_stats(
        merged_df=merged_df,
        individual_stats_dfs=individual_stats_dfs,
        total_initial_cash=total_initial_cash,
        interval=results[0].interval,
        risk_free_rate_annual=risk_free_rate_annual,
        number_of_pairs=len(results),
    )

    final_result = StrategyResult(
        data=merged_df,
        ticker_x="multi",
        ticker_y="pair",
        start=test_start,
        end=test_end,
        interval=results[0].interval,
        fee_rate=results[0].fee_rate,
        window_factor=results[0].window_factor,
        stats=stats,
    )

    save_strategy_result(
        result=final_result,
        file_name=f"test_multi_pair_{test_start}_{test_end}",
        directory=output_dir,
    )

    save_dataframe(
        df=final_result.stats,
        file_name=f"test_stats_multi_pair_{test_start}_{test_end}",
        directory=output_dir,
    )

    btc_data = load_btc_benchmark(
        test_start=test_start,
        test_end=test_end,
        interval=cfg.market.interval,
    )
    plot_returns(final_result, btc_data, directory=output_dir, save=True, show=True)

    logger.info("Merge Multi-Pair Results completed, returning StrategyResult.")

    return final_result
