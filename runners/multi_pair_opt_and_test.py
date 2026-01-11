import logging
import hydra
from omegaconf import OmegaConf, DictConfig
from skopt.space import Real

from modules.performance.strategy import Strategy
from runners.core.pipelines import (
    execute_pair_selection,
    execute_testing,
    setup_run_environment,
    merge_multi_pair_results,
    execute_multi_pair_optimization,
)

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../conf", config_name="base")
def main(cfg: DictConfig):
    output_dir = setup_run_environment(__file__)

    static_params = {}

    if cfg.performance.window == "fixed":
        window_factor_min = cfg.performance.optimization.fixed_window_factor_min
        window_factor_max = cfg.performance.optimization.fixed_window_factor_max
    else:
        window_factor_min = cfg.performance.optimization.window_factor_min
        window_factor_max = cfg.performance.optimization.window_factor_max

    param_space = [
        Real(window_factor_min, window_factor_max, name="window_factor"),
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
    metric = ("objective", "net")

    logger.info(f"Saving results to: {output_dir}")
    logger.info("CONFIG:\n%s", OmegaConf.to_yaml(cfg))

    ps_df = execute_pair_selection(cfg, output_dir)
    logger.info(ps_df)
    selected_pairs_names = ps_df["pair"].tolist()

    strategies = []
    strategies_map = {}

    for pair_name in selected_pairs_names:
        ticker_x, ticker_y = pair_name.split("-")

        bt = Strategy(
            ticker_x,
            ticker_y,
            cfg.performance.optimization.beta_start,
            cfg.performance.test.end,
            cfg.market.interval,
            cfg.market.fee_rate,
            cfg.market.initial_cash,
            cfg.market.risk_free_rate_annual,
            cfg.performance.optimization.min_trades_per_pair,
            cfg.performance.window,
            cfg.performance.source,
            cfg.performance.beta_hedge,
        )
        strategies.append(bt)
        strategies_map[pair_name] = bt

    best_params = execute_multi_pair_optimization(
        cfg, strategies, static_params, param_space, metric
    )

    test_results = []
    test_stats = []

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
            cfg.performance.optimization.beta_start,
            cfg.performance.optimization.start,
            cfg.performance.optimization.end,
        )

        logger.info("--- Starting Test ---")
        result_test = execute_testing(
            cfg,
            bt,
            best_params,
            ticker_x,
            ticker_y,
            output_dir,
            cfg.performance.test.beta_start,
            cfg.performance.test.start,
            cfg.performance.test.end,
        )

        test_results.append(result_test)
        test_stats.append(result_test.stats)

    logger.info("--- Merging Multi-Pair Results ---")

    total_cash_portfolio = cfg.market.initial_cash * len(selected_pairs_names)

    merge_multi_pair_results(
        cfg,
        output_dir,
        test_results,
        test_stats,
        total_cash_portfolio,
        cfg.market.risk_free_rate_annual,
        cfg.performance.test.start,
        cfg.performance.test.end,
    )

if __name__ == "__main__":
    main()
