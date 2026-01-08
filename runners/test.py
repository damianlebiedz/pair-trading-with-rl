import logging
import hydra
from omegaconf import OmegaConf

from modules.performance.strategy import Strategy
from runners.core.pipelines import execute_testing, setup_run_environment

logger = logging.getLogger(__name__)
output_dir = setup_run_environment(__file__)

# =======================================================
ticker_x = "XRPUSDT"
ticker_y = "DOTUSDT"
best_params = {
    "window_factor": 3,
    "entry_threshold": 1.674664587784339,
    "exit_threshold": 0.8039961137276792,
    "stop_loss": 1.5139881477947832,
}
# =======================================================

with hydra.initialize(version_base=None, config_path="../conf"):
    cfg = hydra.compose(config_name="base")

logger.info(f"Saving results to: {output_dir}")
logger.info("CONFIG:\n%s", OmegaConf.to_yaml(cfg))

bt = Strategy(
    ticker_x,
    ticker_y,
    cfg.performance.test.beta_start,
    cfg.performance.test.end,
    cfg.market.interval,
    cfg.market.fee_rate,
    cfg.market.initial_cash,
    cfg.market.risk_free_rate_annual,
    cfg.performance.window,
    cfg.performance.source,
    cfg.performance.beta_hedge,
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
