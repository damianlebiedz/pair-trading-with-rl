import logging
import os
import hydra
from omegaconf import OmegaConf, DictConfig
from skopt.space import Real

from modules.performance.objectives import SortinoWithPenalty
from modules.performance.strategy import Strategy
from runners.core.pipelines import (
    execute_pair_selection,
    execute_testing,
    setup_run_environment,
    merge_multi_pair_results,
    execute_multi_pair_optimization,
    merge_multi_period_results,
)
from runners.core.utils import generate_date_lists

logger = logging.getLogger(__name__)

# =======================================================
static_params = {}
# =======================================================


@hydra.main(version_base=None, config_path="../conf", config_name="base")
def opt_and_test_multi_pair_multi_periods(cfg: DictConfig):
    root = setup_run_environment(__file__)

    config = {
        "pair_selection_opt_start": cfg.pair_selection.optimization.start,
        "pair_selection_test_start": cfg.pair_selection.test.start,
        "pair_selection_opt_end": cfg.pair_selection.optimization.end,
        "pair_selection_test_end": cfg.pair_selection.test.end,
        "opt_beta_start": cfg.performance.optimization.beta_start,
        "opt_start": cfg.performance.optimization.start,
        "opt_end": cfg.performance.optimization.end,
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

        param_space = [
            Real(
                cfg.performance.optimization.window_factor_min,
                cfg.performance.optimization.window_factor_max,
                name="window_factor",
            ),
            Real(
                cfg.performance.optimization.entry_threshold_min,
                cfg.performance.optimization.entry_threshold_max,
                name="entry_threshold",
            ),
            Real(
                cfg.performance.optimization.exit_threshold_min,
                cfg.performance.optimization.exit_threshold_max,
                name="exit_threshold",
            ),
            Real(
                cfg.performance.optimization.stop_loss_min,
                cfg.performance.optimization.stop_loss_max,
                name="stop_loss",
            ),
        ]

        logger.info(f"Saving results to: {output_dir}")
        logger.info("CONFIG:\n%s", OmegaConf.to_yaml(cfg))

        ps_df = execute_pair_selection(
            cfg.tickers,
            lists["pair_selection_test_start_list"][i],
            lists["pair_selection_test_end_list"][i],
            cfg.market.interval,
            cfg.pair_selection.method,
            cfg.pair_selection.eg_factor,
            output_dir,
            lists["pair_selection_opt_start_list"][i],
            lists["pair_selection_opt_end_list"][i],
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
                lists["opt_beta_start_list"][i],
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

        metric_type = "net"
        objective_func = SortinoWithPenalty(
            min_trades_per_pair=cfg.performance.optimization.min_trades_per_pair
        )
        best_params = execute_multi_pair_optimization(
            cfg, strategies, static_params, param_space, metric_type, objective_func
        )

        test_results = []

        logger.info("--- Running Tests with Optimized Parameters ---")

        for pair_name in selected_pairs_names:
            ticker_x, ticker_y = pair_name.split("-")
            bt = strategies_map[pair_name]

            logger.info(f"Testing pair: {pair_name}")

            logger.info("--- Starting Test of Optimization ---")
            execute_testing(
                cfg,
                bt,
                best_params,
                ticker_x,
                ticker_y,
                output_dir,
                lists["opt_beta_start_list"][i],
                lists["opt_start_list"][i],
                lists["opt_end_list"][i],
                "opt",
            )

            logger.info("--- Starting Test ---")
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

        logger.info("--- Merging Multi-Pair Results ---")

        merge_multi_pair_results(
            cfg,
            output_dir,
            test_results,
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
    opt_and_test_multi_pair_multi_periods()
