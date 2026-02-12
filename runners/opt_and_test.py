import logging
import hydra
from omegaconf import OmegaConf, DictConfig
from skopt.space import Real

from modules.performance.strategy import Strategy
from runners.core.pipelines import (
    execute_optimization,
    execute_testing,
    setup_run_environment,
)

logger = logging.getLogger(__name__)

# =======================================================
ticker_x = "BTCUSDT"
ticker_y = "LINKUSDT"

static_params = {}
# =======================================================


@hydra.main(version_base=None, config_path="../conf", config_name="base")
def opt_and_test(cfg: DictConfig):
    output_dir = setup_run_environment(__file__)

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

    logger.info(f"Saving results to: {output_dir}")
    logger.info("CONFIG:\n%s", OmegaConf.to_yaml(cfg))

    bt = Strategy(
        ticker_x=ticker_x,
        ticker_y=ticker_y,
        start=cfg.performance.optimization.beta_start,
        end=cfg.performance.test.end,
        interval=cfg.market.interval,
        fee_rate=cfg.market.fee_rate,
        initial_cash=cfg.market.initial_cash,
        risk_free_rate_annual=cfg.market.risk_free_rate_annual,
        min_trades_per_pair=cfg.performance.optimization.min_trades_per_pair,
        beta_method=cfg.performance.beta_method,
        delayed_entry=cfg.performance.delayed_entry,
        time_decay_sl=(cfg.performance.time_decay_start, cfg.performance.time_decay_end),
    )

    logger.info("--- Starting Optimization ---")

    best_params = execute_optimization(
        cfg=cfg,
        bt=bt,
        static_params=static_params,
        param_space=param_space,
        metric_type=cfg.performance.optimization.metric_type,
        objective_func=cfg.performance.optimization.objective_func,
    )

    logger.info("--- Starting Test of Optimization ---")
    execute_testing(
        cfg=cfg,
        bt=bt,
        best_params=best_params,
        ticker_x=ticker_x,
        ticker_y=ticker_y,
        output_dir=output_dir,
        win_test_start=cfg.performance.optimization.beta_start,
        test_start=cfg.performance.optimization.start,
        test_end=cfg.performance.optimization.end,
        subdir="opt",
    )

    logger.info("--- Starting Test ---")
    execute_testing(
        cfg=cfg,
        bt=bt,
        best_params=best_params,
        ticker_x=ticker_x,
        ticker_y=ticker_y,
        output_dir=output_dir,
        win_test_start=cfg.performance.test.beta_start,
        test_start=cfg.performance.test.start,
        test_end=cfg.performance.test.end,
        subdir="test",
    )


if __name__ == "__main__":
    opt_and_test()
