import logging
import hydra
from omegaconf import OmegaConf
from skopt.space import Real

from modules.performance.strategy import Strategy
from runners.core.pipelines import (
    execute_optimization,
    execute_testing,
    setup_run_environment,
)

logger = logging.getLogger(__name__)
output_dir = setup_run_environment(__file__)

# =======================================================
ticker_x = "BNBUSDT"
ticker_y = "UNIUSDT"
static_params = {
    "window_factor": 1,
    "stop_loss": 2,
}
param_space = [
    Real(1.00, 3.00, name="window_factor"),
    Real(1.10, 4.00, name="entry_threshold"),
    Real(0.0, 1.00, name="exit_threshold"),
    Real(1.10, 2.00, name="stop_loss"),
]
metric = ("objective", "net")
# =======================================================

with hydra.initialize(version_base=None, config_path="../conf"):
    cfg = hydra.compose(config_name="base")

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
    cfg.performance.window,
    cfg.performance.source,
    cfg.performance.beta_hedge,
)

logger.info("--- Starting Optimization ---")
best_params = execute_optimization(cfg, bt, static_params, param_space, metric)

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
