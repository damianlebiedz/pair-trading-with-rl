import logging
import hydra
from omegaconf import OmegaConf, DictConfig

from modules.performance.strategy import Strategy
from runners.core.pipelines import execute_testing, setup_run_environment

logger = logging.getLogger(__name__)

# =======================================================
ticker_x = "BTCUSDT"
ticker_y = "XRPUSDT"
best_params = {
    "window_factor": 2,
    "entry_threshold": 2,
    "exit_threshold": 0,
    "stop_loss": 2,
}
# =======================================================


@hydra.main(version_base=None, config_path="../conf", config_name="base")
def test(cfg: DictConfig):
    output_dir = setup_run_environment(__file__)

    logger.info(f"Saving results to: {output_dir}")
    logger.info("CONFIG:\n%s", OmegaConf.to_yaml(cfg))

    bt = Strategy(
        ticker_x,
        ticker_y,
        # cfg.performance.test.beta_start,
        # cfg.performance.test.end,
        "2024-10-01",
        "2024-12-01",
        cfg.market.interval,
        cfg.market.fee_rate,
        cfg.market.initial_cash,
        cfg.market.risk_free_rate_annual,
        cfg.performance.optimization.min_trades_per_pair,
        cfg.performance.window,
        cfg.performance.beta_hedge,
    )

    logger.info("--- Starting Test ---")
    execute_testing(
        cfg,
        bt,
        best_params,
        ticker_x,
        ticker_y,
        output_dir,
        # cfg.performance.test.beta_start,
        # cfg.performance.test.start,
        # cfg.performance.test.end,
        "2024-10-01",
        "2024-11-01",
        "2024-12-01",
        subdir="test",
    )


if __name__ == "__main__":
    test()
