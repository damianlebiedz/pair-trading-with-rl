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
        cfg.tickers,
        cfg.pair_selection.test.start,
        cfg.pair_selection.test.end,
        cfg.market.interval,
        cfg.pair_selection.method,
        cfg.pair_selection.ps_factor,
        cfg.pair_selection.top_n_factor,
        output_dir,
        cfg.pair_selection.coint_type,
        cfg.pair_selection.optimization.start,
        cfg.pair_selection.optimization.end,
    )

    logger.info(f"\n{df}")


if __name__ == "__main__":
    pair_selection()
