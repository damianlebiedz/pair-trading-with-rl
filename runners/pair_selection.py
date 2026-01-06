import logging
import hydra
from omegaconf import OmegaConf

from runners.core.pipelines import execute_pair_selection, setup_run_environment

logger = logging.getLogger(__name__)
output_dir = setup_run_environment(__file__)

with hydra.initialize(version_base=None, config_path="../conf"):
    cfg = hydra.compose(config_name="base")

logger.info(f"Saving results to: {output_dir}")
logger.info("CONFIG:\n%s", OmegaConf.to_yaml(cfg))

df = execute_pair_selection(cfg, output_dir)
