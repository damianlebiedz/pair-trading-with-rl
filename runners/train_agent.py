import logging
from pathlib import Path
import hydra
import numpy as np
import wandb
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf
from stable_baselines3 import A2C
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import VecNormalize
import os
from wandb.integration.sb3 import WandbCallback

from modules.data_services.data_utils import load_strategy_result
from modules.learning.environments import build_multi_env
from runners.core.pipelines import setup_rl_run_environment
from runners.core.utils import save_hydra_config_snapshot

logger = logging.getLogger(__name__)
load_dotenv()

ALGO_MAP = {"a2c_baseline": A2C, "recurrent_ppo": RecurrentPPO}


class LogEquityCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(LogEquityCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])

        if infos:
            avg_equity = np.mean([info.get("equity", 0.0) for info in infos])
            active_positions = sum(
                [1 for info in infos if abs(info.get("position", 0)) > 0.1]
            )

            if wandb.run is not None:
                wandb.log(
                    {
                        "env/avg_equity": avg_equity,
                        "env/active_positions": active_positions,
                        "global_step": self.num_timesteps,
                    }
                )

        return True


@hydra.main(version_base=None, config_path="../config", config_name="train_agent")
def train_agent(cfg: DictConfig):
    rl_root = setup_rl_run_environment(__file__)
    save_hydra_config_snapshot(cfg=cfg, root_dir=rl_root)

    seed = cfg.rl.seed
    set_random_seed(seed)
    logger.info(f"Random seed set to: {seed}")

    data_path = os.path.join(rl_root, "training_data")
    model_dir = os.path.join(rl_root, "models")
    log_dir = os.path.join(rl_root, "tensorboard_logs")

    os.makedirs(data_path, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    run = wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        group=cfg.wandb.group,
        mode=cfg.wandb.mode,
        config=OmegaConf.to_container(cfg, resolve=True),
        sync_tensorboard=True,
        monitor_gym=True,
        save_code=True,
    )

    logger.info(f"Loading training data from '{data_path}'...")

    all_files = list(Path(data_path).rglob("*.parquet"))
    valid_files = [
        f for f in all_files
        if f.name.startswith("returns_") and "multi_pair" not in f.name
    ]

    if not valid_files:
        logger.error("Parquets not found")
        wandb.finish()
        return

    logger.info(f"Found {len(valid_files)} parquets")

    results = []
    for file_path in valid_files:
        try:
            res = load_strategy_result(file_path.name, directory=str(file_path.parent))
            results.append(res)
        except Exception as e:
            logger.warning(f"Error loading {file_path.name}: {e}")

    if not results:
        logger.error("Data not found")
        wandb.finish()
        return

    logger.info(f"Successfully loaded {len(results)} environments")

    if wandb.run is not None:
        data_artifact = wandb.Artifact(name="training_dataset_multi", type="dataset")
        data_artifact.add_dir(data_path)
        wandb.log_artifact(data_artifact)

    vec_env = build_multi_env(
        results=results,
        rl_reward=cfg.rl.reward,
        obs_space_type=cfg.rl.obs_space_type,
        seed=seed,
    )
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    algo_name = cfg.rl_algo.algo_name
    algo_class = ALGO_MAP.get(algo_name)

    if algo_class is None:
        raise ValueError(f"RL algorithm not found: {algo_name}")

    model_params = OmegaConf.to_container(cfg.rl_algo.params, resolve=True)

    model = algo_class(
        policy=cfg.policy_type,
        env=vec_env,
        verbose=cfg.rl.verbose,
        tensorboard_log=log_dir,
        seed=seed,
        **model_params,
    )

    logger.info(
        f"Starting {algo_name} training on {len(results)} pairs (Run ID: {run.id})..."
    )

    callbacks = [
        WandbCallback(
            gradient_save_freq=100,
            model_save_path=f"{model_dir}/{run.id}",
            verbose=2,
        ),
        LogEquityCallback(),
    ]

    try:
        total_rows = sum(len(res.data) for res in results)
        passes = cfg.rl.passes_per_pair
        calculated_timesteps = total_rows * passes

        model.learn(total_timesteps=calculated_timesteps, callback=callbacks)
        logger.info("Training finished.")

        final_model_name = f"{algo_name}_{cfg.rl.obs_space_type}_{run.id}_seed{seed}"
        save_path = f"{model_dir}/{final_model_name}"
        model.save(save_path)
        vec_env.save(f"{save_path}_normalize.pkl")

        if wandb.run is not None:
            model_artifact = wandb.Artifact(
                name=f"{algo_name}_{cfg.rl.obs_space_type}_model_{run.id}",
                type="model",
                description=f"Trained {algo_name} model (Space: {cfg.rl.obs_space_type})",
            )
            model_artifact.add_file(f"{save_path}.zip")
            model_artifact.add_file(f"{save_path}_normalize.pkl")
            wandb.log_artifact(model_artifact)
            logger.info("Logged model artifact to W&B.")

    except KeyboardInterrupt:
        logger.info("Training interrupted manually. Saving current model...")
        final_model_name = (
            f"{algo_name}_{cfg.rl.obs_space_type}_{run.id}_seed{seed}_interrupted"
        )
        save_path = f"{model_dir}/{final_model_name}"
        model.save(save_path)
        vec_env.save(f"{save_path}_normalize.pkl")

        if wandb.run is not None:
            model_artifact = wandb.Artifact(
                name=f"{algo_name}_{cfg.rl.obs_space_type}_model_{run.id}_interrupted",
                type="model",
                description=f"Interrupted {algo_name} (Space: {cfg.rl.obs_space_type})",
            )
            model_artifact.add_file(f"{save_path}.zip")
            model_artifact.add_file(f"{save_path}_normalize.pkl")
            wandb.log_artifact(model_artifact)
            logger.info("Saved interrupted model.")
    finally:
        wandb.finish()


if __name__ == "__main__":
    train_agent()
