import logging
import os
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Literal
import pandas as pd

from modules.performance.models import StrategyResult
from modules.data_services.data_utils import (
    save_strategy_result,
    load_btc_benchmark,
    save_dataframe,
    load_strategy_result,
    load_ewp_benchmark,
)
from modules.performance.multi_pair_optimizer import MultiPairOptimizer
from modules.data_services.merge_utils import (
    aggregate_strategy_results,
    stitch_strategy_results,
)
from modules.performance.objectives import SortinoWithPenalty
from modules.performance.pair_selector import PairSelector
from modules.performance.stats import calculate_stats
from modules.performance.strategy import Strategy
from modules.utils.plots import plot_returns, plot_zscore_pos, plot_spread_pos

logger = logging.getLogger(__name__)


def setup_run_environment(calling_file: str) -> str:
    script_dir = os.path.dirname(os.path.abspath(calling_file))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))

    file_stem = Path(calling_file).stem
    unique_id = uuid.uuid4().hex[:6]
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f"_{unique_id}"
    output_dir = os.path.join(project_root, "results", f"{file_stem}_{timestamp}")

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

    data_dir = os.path.join(project_root, "data_rl")
    models_dir = os.path.join(data_dir, "models")
    training_data_dir = os.path.join(data_dir, "training_data")

    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(training_data_dir, exist_ok=True)

    logger.debug("--- RL Environment Setup ---")
    logger.debug(f"Directory: {data_dir}")

    return data_dir


def execute_testing(
    bt: Strategy,
    best_params: dict[str, Any],
    ticker_x: str,
    ticker_y: str,
    output_dir: str,
    win_test_start: str,
    test_start: str,
    test_end: str,
    interval: str,
    plot: bool,
    tickers: list[str],
    subdir: str | None = None,
) -> StrategyResult:

    if subdir:
        output_dir = os.path.join(output_dir, subdir)

    fixed_window = best_params["fixed_window"]
    entry_threshold = best_params["entry_threshold"]
    exit_threshold = best_params["exit_threshold"]
    stop_loss = best_params["stop_loss"]

    result = bt.run_strategy(
        fixed_window=fixed_window,
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

    if plot:
        plot_zscore_pos(result, directory=output_dir, save=True)
        plot_spread_pos(result, directory=output_dir, save=True)
        btc_data = load_btc_benchmark(
            test_start=test_start,
            test_end=test_end,
            interval=interval,
        )
        ewp_data = load_ewp_benchmark(
            tickers=tickers,
            test_start=test_start,
            test_end=test_end,
            interval=interval,
        )
        plot_returns(
            result=result,
            btc_data=btc_data,
            ewp_data=ewp_data,
            directory=output_dir,
            save=True,
        )

        logger.debug("Testing completed, returning StrategyResult.")

    return result


def execute_pair_selection(
    tickers: list[str],
    ps_start: str,
    ps_end: str,
    test_win_start: str,
    interval: str,
    top_n_factor: float,
    output_dir: str,
    coint_type: Literal["eg", "johansen"],
    beta_method: Literal["ols", "kalman"],
    valid_window: tuple[int, int],
) -> pd.DataFrame:
    logger.info("Starting Pair Selection Pipeline.")

    selector = PairSelector(
        coint_type=coint_type, beta_method=beta_method, valid_window=valid_window
    )

    final_df = selector.select_pairs(
        tickers=tickers,
        ps_start=ps_start,
        ps_end=ps_end,
        test_win_start=test_win_start,
        interval=interval,
        top_n=top_n_factor,
    )

    if final_df.empty:
        logger.warning("No pairs selected.")
        return pd.DataFrame()

    save_dataframe(
        df=final_df,
        file_name=f"pair_selection_{coint_type}_{ps_start}_{ps_end}",
        directory=output_dir,
    )

    logger.info(f"Pair Selection completed. Saved {len(final_df)} pairs.")

    return final_df


def execute_multi_pair_optimization(
    strategies: list[Strategy],
    static_params: dict[str, Any],
    param_space: list[Any],
    metric_type: Literal["gross", "net"],
    objective_func: Literal["sortino"],
    opt_start: str,
    opt_end: str,
    opt_win_start: str,
    penalty_bad: float,
    n_iter: int,
    interval: str,
    risk_free_rate_annual: float,
    min_trades_per_pair: int,
    initial_cash: float,
) -> dict[str, Any]:
    logger.info(f"Starting Multi-Pair Optimization on {len(strategies)} pairs...")

    if objective_func not in ["sortino"]:
        raise ValueError("objective_func should be 'sortino'")

    if objective_func == "sortino":
        objective_func = SortinoWithPenalty(
            min_trades_per_pair=min_trades_per_pair,
            penalty_bad=penalty_bad,
        )

    optimizer = MultiPairOptimizer(
        strategies=strategies,
        opt_start=opt_start,
        opt_end=opt_end,
        opt_win_start=opt_win_start,
        penalty_bad=penalty_bad,
        n_iter=n_iter,
        interval=interval,
        risk_free_rate_annual=risk_free_rate_annual,
        min_trades_per_pair=min_trades_per_pair,
        initial_cash=initial_cash,
        number_of_pairs=len(strategies),
    )

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


def merge_multi_pair_results(
    output_dir: str,
    results: list[StrategyResult],
    initial_cash: float,
    risk_free_rate_annual: float,
    test_start: str,
    test_end: str,
    interval: str,
    plot: bool,
    tickers: list[str],
    prefix: str | None = "",
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
        file_name=f"{prefix}returns_multi_pair_{test_start}_{test_end}",
        directory=output_dir,
    )

    save_dataframe(
        df=merged_exec_res,
        file_name=f"{prefix}exec_logger_multi_pair_{final_result.start}_{final_result.end}",
        directory=output_dir,
    )

    save_dataframe(
        df=final_result.stats,
        file_name=f"{prefix}stats_multi_pair_{test_start}_{test_end}",
        directory=output_dir,
    )

    if plot:
        btc_data = load_btc_benchmark(
            test_start=final_result.start,
            test_end=final_result.end,
            interval=interval,
        )
        ewp_data = load_ewp_benchmark(
            tickers=tickers,
            test_start=final_result.start,
            test_end=final_result.end,
            interval=interval,
        )
        plot_returns(
            result=final_result,
            btc_data=btc_data,
            ewp_data=ewp_data,
            directory=output_dir,
            save=True,
            prefix=prefix,
        )

    logger.debug("Merge Multi-Pair Results completed, returning StrategyResult.")

    return final_result


def merge_multi_period_results(
    output_dir: str,
    ticker_x: str,
    ticker_y: str,
    initial_cash: float,
    risk_free_rate_annual: float,
    interval: str,
    plot: bool,
    tickers: list[str],
    prefix: str = "",
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

    logger.debug(
        f"Detected multi-period structure in {output_dir}. Merging results for {ticker_x}-{ticker_y}..."
    )

    iter_dirs = sorted(
        [d for d in base_path.iterdir() if d.is_dir() and d.name.isdigit()],
        key=lambda x: int(x.name),
    )

    results = []

    for d in iter_dirs:
        pattern = f"{prefix}returns_{ticker_x}_{ticker_y}_*.parquet"
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
        file_name=f"{prefix}returns_{ticker_x}_{ticker_y}_{final_result.start}_{final_result.end}",
        directory=output_dir,
    )

    save_dataframe(
        df=final_exec_df,
        file_name=f"{prefix}exec_logger_{ticker_x}_{ticker_y}_{final_result.start}_{final_result.end}",
        directory=output_dir,
    )

    save_dataframe(
        df=final_result.stats,
        file_name=f"{prefix}stats_{ticker_x}_{ticker_y}_{final_result.start}_{final_result.end}",
        directory=output_dir,
    )

    if plot:
        btc_data = load_btc_benchmark(
            test_start=final_result.start,
            test_end=final_result.end,
            interval=interval,
        )
        ewp_data = load_ewp_benchmark(
            tickers=tickers,
            test_start=final_result.start,
            test_end=final_result.end,
            interval=interval,
        )
        plot_returns(
            result=final_result,
            btc_data=btc_data,
            ewp_data=ewp_data,
            directory=output_dir,
            save=True,
            prefix=prefix,
        )

    logger.debug(f"Merge Multi-Period Results for {ticker_x}-{ticker_y} completed.")

    return final_result
