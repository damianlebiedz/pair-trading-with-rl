import logging
import os
import hydra
from omegaconf import OmegaConf, DictConfig

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
    "window_factor": 2,
    "entry_threshold": 2,
    "exit_threshold": 0,
    "stop_loss": 2,
}
# =======================================================


@hydra.main(version_base=None, config_path="../conf", config_name="base")
def test_multi_pair_multi_periods(cfg: DictConfig):
    root = setup_run_environment(__file__)

    config = {
        "pair_selection_test_start": cfg.pair_selection.test.start,
        "pair_selection_test_end": cfg.pair_selection.test.end,
        "test_beta_start": cfg.performance.test.beta_start,
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
            cfg.tickers,
            lists["pair_selection_test_start_list"][i],
            lists["pair_selection_test_end_list"][i],
            cfg.market.interval,
            cfg.pair_selection.method,
            cfg.pair_selection.eg_factor,
            output_dir,
        )

        logger.info(ps_df)
        selected_pairs_names = ps_df["pair"].tolist()

        strategies = []
        strategies_map = {}

        for pair_name in selected_pairs_names:
            ticker_x, ticker_y = pair_name.split("-")

            bt = Strategy(
                ticker_x,
                ticker_y,
                lists["test_beta_start_list"][i],
                lists["test_end_list"][i],
                cfg.market.interval,
                cfg.market.fee_rate,
                cfg.market.initial_cash / len(selected_pairs_names),
                cfg.market.risk_free_rate_annual,
                cfg.performance.optimization.min_trades_per_pair,
                cfg.performance.window,
                cfg.performance.beta_hedge,
            )
            strategies.append(bt)
            strategies_map[pair_name] = bt

        test_results = []
        test_stats = []

        logger.info("--- Running Tests with Optimized Parameters ---")

        for pair_name in selected_pairs_names:
            ticker_x, ticker_y = pair_name.split("-")
            bt = strategies_map[pair_name]

            logger.info(f"--- Testing pair: {pair_name} ---")

            result_test = execute_testing(
                cfg,
                bt,
                best_params,
                ticker_x,
                ticker_y,
                output_dir,
                lists["test_beta_start_list"][i],
                lists["test_start_list"][i],
                lists["test_end_list"][i],
                "test",
            )

            test_results.append(result_test)
            test_stats.append(result_test.stats)

        logger.info("--- Merging Multi-Pair Results ---")

        merge_multi_pair_results(
            cfg,
            output_dir,
            test_results,
            test_stats,
            cfg.market.initial_cash,
            cfg.market.risk_free_rate_annual,
            lists["test_start_list"][i],
            lists["test_end_list"][i],
        )

    merge_multi_period_results(
        cfg,
        root,
        "multi",
        "pair",
        cfg.market.initial_cash,
        cfg.market.risk_free_rate_annual,
    )


if __name__ == "__main__":
    test_multi_pair_multi_periods()
