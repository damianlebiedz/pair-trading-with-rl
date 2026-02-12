import logging
import hydra
from omegaconf import OmegaConf, DictConfig
from skopt.space import Real

from modules.performance.objectives import SortinoWithPenalty
from modules.performance.strategy import Strategy
from runners.core.pipelines import (
    execute_optimization,
    execute_testing,
    setup_run_environment,
)

logger = logging.getLogger(__name__)

# =======================================================
ticker_x = "BNBUSDT"
ticker_y = "UNIUSDT"

static_params = {}
# =======================================================


@hydra.main(version_base=None, config_path="../conf", config_name="base")
def opt(cfg: DictConfig):
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
        ticker_x,
        ticker_y,
        cfg.performance.optimization.beta_start,
        cfg.performance.optimization.end,
        cfg.market.interval,
        cfg.market.fee_rate,
        cfg.market.initial_cash,
        cfg.market.risk_free_rate_annual,
        cfg.performance.optimization.min_trades_per_pair,
        cfg.performance.window,
        cfg.performance.beta_hedge,
    )

    logger.info("--- Starting Optimization ---")

    metric_type = "net"
    objective_func = SortinoWithPenalty(
        min_trades_per_pair=cfg.performance.optimization.min_trades_per_pair
    )
    best_params = execute_optimization(
        cfg, bt, static_params, param_space, metric_type, objective_func
    )

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
        subdir="opt",
    )


if __name__ == "__main__":
    opt()
