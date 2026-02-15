import logging
import hydra
from omegaconf import OmegaConf, DictConfig

from runners.core.pipelines import execute_pair_selection, setup_run_environment

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../conf", config_name="base")
def pair_selection(cfg: DictConfig):
    output_dir = setup_run_environment(__file__)

    logger.info(f"Saving results to: {output_dir}")
    logger.info("CONFIG:\n%s", OmegaConf.to_yaml(cfg))

    df = execute_pair_selection(
        tickers=cfg.tickers,
        ps_start=cfg.pair_selection.start,
        ps_end=cfg.pair_selection.end,
        test_win_start=cfg.pair_selection.test.start,
        interval=cfg.market.interval,
        top_n_factor=cfg.pair_selection.top_n_factor,
        output_dir=output_dir,
        coint_type=cfg.pair_selection.coint_type,
        beta_method=cfg.performance.beta_method,
        valid_window=(cfg.performance.window_min, cfg.performance.window_max),
    )

    logger.info(f"\n{df}")


if __name__ == "__main__":
    pair_selection()
