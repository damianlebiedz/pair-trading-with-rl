import logging
import hydra
from omegaconf import OmegaConf, DictConfig

from modules.performance.strategy import Strategy
from runners.core.pipelines import execute_testing, setup_run_environment

logger = logging.getLogger(__name__)

# =======================================================
ticker_x = "AVAXUSDT"
ticker_y = "OPUSDT"
best_params = {
    "window_factor": 1,
    "entry_threshold": 2.5,
    "exit_threshold": 0.2,
    "stop_loss": 2,
}
# =======================================================


@hydra.main(version_base=None, config_path="../conf", config_name="base")
def test(cfg: DictConfig):
    output_dir = setup_run_environment(__file__)

    logger.info(f"Saving results to: {output_dir}")
    logger.info("CONFIG:\n%s", OmegaConf.to_yaml(cfg))

    bt = Strategy(
        ticker_x=ticker_x,
        ticker_y=ticker_y,
        start=cfg.performance.test.win_start,
        end=cfg.performance.test.end,
        interval=cfg.market.interval,
        fee_rate=cfg.market.fee_rate,
        initial_cash=cfg.market.initial_cash,
        risk_free_rate_annual=cfg.market.risk_free_rate_annual,
        min_trades_per_pair=cfg.performance.optimization.min_trades_per_pair,
        beta_method=cfg.performance.beta_method,
        delayed_entry=cfg.performance.delayed_entry,
        time_decay_sl=(
            cfg.performance.time_decay_start,
            cfg.performance.time_decay_end,
        ),
    )

    logger.info("--- Starting Test ---")
    execute_testing(
        cfg=cfg,
        bt=bt,
        best_params=best_params,
        ticker_x=ticker_x,
        ticker_y=ticker_y,
        output_dir=output_dir,
        win_test_start=cfg.performance.test.win_start,
        test_start=cfg.performance.test.start,
        test_end=cfg.performance.test.end,
        subdir="test",
    )


if __name__ == "__main__":
    test()
