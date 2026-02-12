import logging
import os
import hydra
import pandas as pd
from omegaconf import OmegaConf, DictConfig

from modules.core.models import StrategyResult
from modules.data_services.data_loaders import load_data
from modules.data_services.data_utils import save_dataframe, save_strategy_result
from modules.performance.strategy import Strategy
from runners.core.pipelines import (
    execute_pair_selection,
    execute_testing,
    setup_run_environment,
    merge_multi_pair_results,
    merge_multi_period_results,
)
from runners.core.utils import generate_date_lists

logger = logging.getLogger(__name__)

# =======================================================
best_params = {
    "window_factor": 1,
    "entry_threshold": 2.5,
    "exit_threshold": 0.2,
    "stop_loss": 2,
}
# =======================================================


@hydra.main(version_base=None, config_path="../conf", config_name="base")
def test_multi_pair(cfg: DictConfig):
    root = setup_run_environment(__file__)

    config = {
        "pair_selection_test_start": cfg.pair_selection.test.start,
        "pair_selection_test_end": cfg.pair_selection.test.end,
        "test_win_start": cfg.performance.test.win_start,
        "test_start": cfg.performance.test.start,
        "test_end": cfg.performance.test.end,
    }

    number_of_iterations = cfg.performance.iterations
    lists = generate_date_lists(config, number_of_iterations)

    logger.info(f"Saving results to: {root}")
    logger.info("CONFIG:\n%s", OmegaConf.to_yaml(cfg))

    for i in range(number_of_iterations):
        output_dir = os.path.join(root, f"{i+1}")
        if number_of_iterations == 1:
            output_dir = root

        logger.info(f"--- Running Iteration {i+1} ---")

        ps_df = execute_pair_selection(
            tickers=cfg.tickers,
            test_start=lists["pair_selection_test_start_list"][i],
            test_end=lists["pair_selection_test_end_list"][i],
            interval=cfg.market.interval,
            method=cfg.pair_selection.method,
            ps_factor=cfg.pair_selection.ps_factor,
            top_n_factor=cfg.pair_selection.top_n_factor,
            output_dir=output_dir,
            coint_type=cfg.pair_selection.coint_type,
        )

        logger.info(f"\n{ps_df}")
        selected_pairs_names = ps_df["pair"].tolist()

        if not selected_pairs_names:
            logger.warning(
                f"Iteration {i + 1}: No pairs selected! Generating flat (cash-only) result for this period."
            )

            ref_ticker = cfg.tickers[0]
            ref_data = load_data(
                tickers=[ref_ticker],
                start=lists["test_start_list"][i],
                end=lists["test_end_list"][i],
                interval=cfg.market.interval,
            )

            empty_data = pd.DataFrame(index=ref_data.index)
            empty_data["total_pnl"] = 0.0
            empty_data["total_net_pnl"] = 0.0
            empty_data["total_return"] = 0.0
            empty_data["total_net_return"] = 0.0
            empty_data["in_position"] = 0.0

            empty_result = StrategyResult(
                data=empty_data,
                ticker_x="multi",
                ticker_y="pair",
                start=lists["test_start_list"][i],
                end=lists["test_end_list"][i],
                interval=cfg.market.interval,
                fee_rate=cfg.market.fee_rate,
                stats=pd.DataFrame(),
                exec_logger=pd.DataFrame(),
            )

            save_strategy_result(
                result=empty_result,
                file_name=f"returns_multi_pair_{empty_result.start}_{empty_result.end}",
                directory=output_dir,
            )

            save_dataframe(
                df=pd.DataFrame(),
                file_name=f"exec_logger_multi_pair_{empty_result.start}_{empty_result.end}",
                directory=output_dir,
            )

            save_dataframe(
                df=pd.DataFrame(),
                file_name=f"stats_multi_pair_{empty_result.start}_{empty_result.end}",
                directory=output_dir,
            )

            continue

        strategies = []
        strategies_map = {}

        for pair_name in selected_pairs_names:
            ticker_x, ticker_y = pair_name.split("-")

            bt = Strategy(
                ticker_x=ticker_x,
                ticker_y=ticker_y,
                start=lists["test_win_start_list"][i],
                end=lists["test_end_list"][i],
                interval=cfg.market.interval,
                fee_rate=cfg.market.fee_rate,
                initial_cash=cfg.market.initial_cash / len(selected_pairs_names),
                risk_free_rate_annual=cfg.market.risk_free_rate_annual,
                min_trades_per_pair=cfg.performance.optimization.min_trades_per_pair,
                beta_method=cfg.performance.beta_method,
                delayed_entry=cfg.performance.delayed_entry,
                time_decay_sl=(cfg.performance.time_decay_start, cfg.performance.time_decay_end),
            )

            strategies.append(bt)
            strategies_map[pair_name] = bt

        test_results = []

        for pair_name in selected_pairs_names:
            ticker_x, ticker_y = pair_name.split("-")
            bt = strategies_map[pair_name]

            logger.info(f"--- Testing pair: {pair_name} ---")

            result_test = execute_testing(
                cfg=cfg,
                bt=bt,
                best_params=best_params,
                ticker_x=ticker_x,
                ticker_y=ticker_y,
                output_dir=output_dir,
                win_test_start=lists["test_win_start_list"][i],
                test_start=lists["test_start_list"][i],
                test_end=lists["test_end_list"][i],
                subdir="test",
            )

            test_results.append(result_test)

        logger.info("--- Merging Multi-Pair Results ---")

        merge_multi_pair_results(
            cfg=cfg,
            output_dir=output_dir,
            results=test_results,
            initial_cash=cfg.market.initial_cash,
            risk_free_rate_annual=cfg.market.risk_free_rate_annual,
            test_start=lists["test_start_list"][i],
            test_end=lists["test_end_list"][i],
        )

    merge_multi_period_results(
        cfg=cfg,
        output_dir=root,
        ticker_x="multi",
        ticker_y="pair",
        initial_cash=cfg.market.initial_cash,
        risk_free_rate_annual=cfg.market.risk_free_rate_annual,
    )


if __name__ == "__main__":
    test_multi_pair()
