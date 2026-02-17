import logging
import os

import hydra
from omegaconf import OmegaConf, DictConfig

from modules.learning.agents import RLAgentAdapter
from modules.performance.strategy import Strategy
from runners.core.pipelines import (
    execute_testing,
    setup_run_environment,
    setup_rl_run_environment,
)
from runners.core.utils import load_model

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../config", config_name="test")
def test(cfg: DictConfig):
    output_dir = setup_run_environment(__file__)

    ticker_x = cfg.ticker_x
    ticker_y = cfg.ticker_y

    best_params = {
        "fixed_window": cfg.fixed_window,
        "entry_threshold": cfg.entry_threshold,
        "exit_threshold": cfg.exit_threshold,
        "stop_loss": cfg.stop_loss,
    }

    rl_output_dir = None
    if cfg.performance.rl:
        rl_output_dir = setup_rl_run_environment(__file__)

    logger.info(f"Saving results to: {output_dir}")
    logger.info("CONFIG:\n%s", OmegaConf.to_yaml(cfg))

    agent = None
    if cfg.performance.rl:
        model_path = os.path.join(rl_output_dir, "models")
        try:
            model = load_model(path=model_path)
            agent = RLAgentAdapter(model=model, training_mode=False)
            logger.info("RL Agent loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load RL model: {e}")

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
        beta_hedge=cfg.performance.beta_hedge,
        beta_method=cfg.performance.beta_method,
        window_method=cfg.performance.window_method,
        delayed_entry=cfg.performance.delayed_entry,
        sl_lock=cfg.performance.sl_lock,
        time_decay_sl=(
            cfg.performance.time_decay_start,
            cfg.performance.time_decay_end,
        ),
        valid_window=(cfg.performance.window_min, cfg.performance.window_max),
        vol_window=cfg.performance.vol_window,
        agent=agent,
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
