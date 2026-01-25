import logging
import os
import sys
from datetime import datetime
from pathlib import Path
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
    load_strategy_result,
    load_dataframe,
)
from modules.multi_pair.multi_pair_optimizer import MultiPairOptimizer
from modules.multi_pair.multi_pair_utils import aggregate_strategy_results
from modules.performance.stats import calculate_multi_pair_stats
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
        beta_test_start=beta_test_start,
    )

    save_strategy_result(
        result=result,
        file_name=f"returns_{ticker_x}_{ticker_y}_{test_start}_{test_end}",
        directory=output_dir,
    )

    plot_zscore_pos(result, directory=output_dir, save=True)
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
    eg_factor: float,
    output_dir: str,
    opt_start: str | None = None,
    opt_end: str | None = None,
) -> pd.DataFrame:
    selection_method = method
    if selection_method not in ["both", "second"]:
        raise ValueError("'method' must be 'both' or 'second'")

    df_test = load_data(
        tickers=tickers,
        start=test_start,
        end=test_end,
        interval=interval,
    )
    eg_df_test = engle_granger_cointegration(df_test)

    if selection_method == "second" or opt_start is None or opt_end is None:
        condition = eg_df_test["eg_p_value"] <= eg_factor
        sort_column = "eg_p_value"
        logger.info("Selection method: 'second' (filtering by test set only).")

        final_df = eg_df_test[condition].sort_values(sort_column).reset_index(drop=True)
        final_df = final_df.round(4)

        start = test_start

    else:
        df_opt = load_data(
            tickers=tickers,
            start=opt_start,
            end=opt_end,
            interval=interval,
        )
        eg_df_opt = engle_granger_cointegration(df_opt)

        merged_df = pd.merge(
            eg_df_opt, eg_df_test, on="pair", how="inner", suffixes=("_opt", "_test")
        )

        condition = (eg_df_test["eg_p_value_opt"] <= eg_factor) & (
            eg_df_test["eg_p_value_test"] <= eg_factor
        )
        sort_column = "eg_p_value_test"
        logger.info(
            "Selection method: 'both' (filtering by optimization AND test sets)."
        )

        final_df = merged_df[condition].sort_values(sort_column).reset_index(drop=True)
        final_df = final_df.round(4)

        start = opt_start

    save_dataframe(
        df=final_df,
        file_name=f"pair_selection_{method}_{start}_{test_end}",
        directory=output_dir,
    )

    logger.info(f"Pair Selection completed. Selected {len(final_df)} pairs.")

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
    """Merges multiple StrategyResult objects into one aggregate result and saves it."""
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
        min_trades_per_pair=cfg.performance.optimization.min_trades_per_pair,
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
    )

    save_strategy_result(
        result=final_result,
        file_name=f"returns_multi_pair_{test_start}_{test_end}",
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
    chronologically in a staircase manner.
    Loads stats from each period individually to aggregate them.
    """
    base_path = Path(output_dir)

    if not (base_path / "1").exists():
        raise ValueError(
            f"Cannot perform multi-period merge when there is no multi periods in {output_dir}"
        )

    logger.info(
        f"Detected multi-period structure in {output_dir}. Merging results for {ticker_x}-{ticker_y}..."
    )

    iter_dirs = sorted(
        [d for d in base_path.iterdir() if d.is_dir() and d.name.isdigit()],
        key=lambda x: int(x.name),
    )

    results = []
    collected_stats = []

    for d in iter_dirs:
        pattern = f"returns_{ticker_x}_{ticker_y}_*.parquet"
        files = list(d.glob(pattern))

        if not files:
            logger.warning(f"No result file found for {ticker_x}-{ticker_y} in {d}")
            continue

        file_path = files[0]
        file_stem = file_path.stem

        res = load_strategy_result(file_stem, directory=str(d))
        results.append(res)

        stats_pattern = f"stats_{ticker_x}_{ticker_y}_*.parquet"
        stats_files = list(d.glob(stats_pattern))

        if not stats_files:
            logger.warning(f"Stats file not found for {ticker_x}-{ticker_y} in {d}")
        else:
            stats_path = stats_files[0]
            stats_filename = stats_path.stem

            stats_df = load_dataframe(stats_filename, directory=str(d))

            if "metric" in stats_df.columns:
                stats_df = stats_df.set_index("metric")

            collected_stats.append(stats_df)

    if not results:
        logger.warning("No results collected for merging.")
        return None

    merged_dfs = []
    offset_return = 0.0
    offset_net = 0.0
    offset_return_pct = 0.0
    offset_net_pct = 0.0

    for res in results:
        df = res.data.copy()

        if "open_time" in df.columns:
            df = df.set_index("open_time")

        if "total_return" in df.columns:
            df["total_return"] += offset_return
        if "net_return" in df.columns:
            df["net_return"] += offset_net
        if "total_return_pct" in df.columns:
            df["total_return_pct"] += offset_return_pct
        if "net_return_pct" in df.columns:
            df["net_return_pct"] += offset_net_pct

        merged_dfs.append(df)

        if not df.empty:
            if "total_return" in df.columns:
                offset_return = df["total_return"].iloc[-1]
            if "net_return" in df.columns:
                offset_net = df["net_return"].iloc[-1]
            if "total_return_pct" in df.columns:
                offset_return_pct = df["total_return_pct"].iloc[-1]
            if "net_return_pct" in df.columns:
                offset_net_pct = df["net_return_pct"].iloc[-1]

    final_df = pd.concat(merged_dfs).sort_index()

    stats = calculate_multi_pair_stats(
        merged_df=final_df,
        individual_stats_dfs=collected_stats,
        total_initial_cash=initial_cash,
        interval=results[0].interval,
        risk_free_rate_annual=risk_free_rate_annual,
        number_of_pairs=0,
        min_trades_per_pair=cfg.performance.optimization.min_trades_per_pair,
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
    )

    save_strategy_result(
        result=final_result,
        file_name=f"returns_{ticker_x}_{ticker_y}_{final_result.start}_{final_result.end}",
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
