import logging
import hydra
from omegaconf import OmegaConf
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
output_dir = setup_run_environment(__file__)

# =======================================================
static_params = {
    # "window_factor": 1,
    # "stop_loss": 2,
}
param_space = [
    Real(1.00, 3.00, name="window_factor"),
    Real(1.10, 3.00, name="entry_threshold"),
    Real(0.0, 1.00, name="exit_threshold"),
    Real(1.10, 2.00, name="stop_loss"),
]
metric = ("objective", "net")
# =======================================================

with hydra.initialize(version_base=None, config_path="../conf"):
    cfg = hydra.compose(config_name="base")

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
        cfg.performance.window,
        cfg.performance.source,
        cfg.performance.beta_hedge,
    )
    strategies.append(bt)
    strategies_map[pair_name] = bt

logger.info("--- Starting Multi-Pair Optimization ---")
best_params = execute_multi_pair_optimization(
    cfg, strategies, static_params, param_space, metric
)

all_results = []
all_stats = []

logger.info("--- Running Tests with Optimized Parameters ---")

for pair_name in selected_pairs_names:
    ticker_x, ticker_y = pair_name.split("-")
    bt = strategies_map[pair_name]

    logger.info(f"Testing pair: {pair_name}")

    logger.info("--- Starting Test of Optimization ---")
    result_opt = execute_testing(
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

    all_results.append(result_test)
    all_stats.append(result_test.stats)

logger.info("--- Merging Multi-Pair Results ---")

total_cash_portfolio = cfg.market.initial_cash * len(selected_pairs_names)

summary = merge_multi_pair_results(
    cfg,
    output_dir,
    all_results,
    all_stats,
    total_cash_portfolio,
    cfg.market.risk_free_rate_annual,
    cfg.performance.test.start,
    cfg.performance.test.end,
)
