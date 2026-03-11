import copy
import logging
import shutil
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

from modules.core.config import Config
from modules.core.enums import RLModelName
from modules.data_services.data_utils import load_strategy_result
from modules.learning.environments import build_multi_env
from runners.core.pipelines import setup_run_environment
from runners.core.utils import save_hydra_config_snapshot, get_first_subdirectory

logger = logging.getLogger(__name__)
load_dotenv()

ALGO_MAP = {
    RLModelName.A2C_BASELINE.value: A2C,
    RLModelName.RECURRENT_PPO.value: RecurrentPPO,
}


class LogEquityCallback(BaseCallback):
    """
    Custom callback for Stable Baselines3 that logs aggregated portfolio statistics to Weights & Biases.
    Logged metrics (visible in the WandB dashboard):

    - portfolio/avg_equity: Average equity across all environments. The primary indicator of whether the overall strategy is profitable.
    - portfolio/exposure_pct: Market exposure (from 0.0 to 1.0). Shows what percentage of pairs have an open position at a given step. Values close to 0 indicate the agent is overly cautious (e.g., due to fees), while values close to 1 suggest overtrading.
    - portfolio/avg_win_rate: Win rate of closed trades. Helps assess whether the agent correctly predicts mean reversion, independent of net profitability.
    - portfolio/avg_hold_time: Average position holding time (in steps/periods). If extremely low (e.g., 1–2 steps), it indicates the agent is panic-closing positions due to negative PnL immediately after entry (spread/fees).
    - portfolio/total_fees_paid: Cumulative total fees paid across the entire portfolio. A critical metric for diagnosing “fee drag” — when exchange fees erode strategy profits.
    """

    def __init__(self, verbose=0):
        super(LogEquityCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])

        if infos:
            avg_equity = np.mean([info.get("equity", 0.0) for info in infos])

            current_positions = [
                1 if abs(i.get("position", 0)) > 0.1 else 0 for i in infos
            ]
            portfolio_exposure = np.mean(current_positions)

            avg_win_rate = np.mean([info.get("win_rate", 0.0) for info in infos])
            avg_hold_time = np.mean([info.get("avg_hold_time", 0.0) for info in infos])
            total_portfolio_fees = np.sum(
                [info.get("ep_total_fees", 0.0) for info in infos]
            )

            if wandb.run is not None:
                wandb.log(
                    {
                        "portfolio/avg_equity": avg_equity,
                        "portfolio/exposure_pct": portfolio_exposure,
                        "portfolio/avg_win_rate": avg_win_rate,
                        "portfolio/avg_hold_time": avg_hold_time,
                        "portfolio/total_fees_paid": total_portfolio_fees,
                        "global_step": self.num_timesteps,
                    }
                )

        return True


@hydra.main(version_base=None, config_path="../config", config_name="train_agent")
def train_agent(cfg: DictConfig):
    root = setup_run_environment(__file__)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))

    training_folder = cfg.rl.training_folder

    if not training_folder:
        training_root = os.path.join(project_root, "data", "rl_training")
        try:
            training_folder = get_first_subdirectory(training_root)
            logger.info(
                f"No training subfolder specified. Auto-selected: {training_folder}"
            )
        except (FileNotFoundError, ValueError) as e:
            logger.error(f"Cannot auto-select training data: {e}")
            return

    training_data_dir = os.path.join(
        project_root, "data", "rl_training", training_folder
    )

    model_dir = os.path.join(project_root, "data", "rl_models")
    log_dir = os.path.join(root, "tensorboard_logs")

    os.makedirs(training_data_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    save_hydra_config_snapshot(cfg=cfg, root_dir=root)

    config_dict = OmegaConf.to_container(cfg, resolve=True)
    cfg = Config(**config_dict)

    seed = cfg.rl.seed
    set_random_seed(seed)
    logger.info(f"Random seed set to: {seed}")

    run_name = f"{cfg.rl_algo.algo_name.value}_{cfg.rl.obs_space_type.value}_{cfg.rl.reward.value}"

    run = wandb.init(
        project=cfg.wandb.project,
        name=run_name,
        mode=cfg.wandb.mode,
        config=config_dict,
        sync_tensorboard=True,
        monitor_gym=True,
        save_code=True,
        dir=root,
    )

    logger.info(f"Loading training data from '{training_data_dir}'...")

    all_files = list(Path(training_data_dir).rglob("*.parquet"))
    valid_files = [
        f
        for f in all_files
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
            res = load_strategy_result(file_path.stem, directory=str(file_path.parent))
            results.append(res)
        except Exception as e:
            logger.warning(f"Error loading {file_path.name}: {e}")

    required_cols = [
        "market_z_score",
        "market_std",
        "market_beta",
        "hurst",
        "window",
        "market_vol",
    ]

    valid_results = []
    MIN_STEPS_REQUIRED = 168

    for res in results:
        df = res.data

        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.warning(
                f"Skipping pair: {res.ticker_x}-{res.ticker_y} - Data not found in: {missing_cols}"
            )
            continue

        is_valid = df[required_cols].notnull().all(axis=1)
        blocks = is_valid.ne(is_valid.shift()).cumsum()

        part_counter = 1
        for block_id, block_df in df.groupby(blocks):
            if is_valid.loc[block_df.index[0]]:

                if len(block_df) >= MIN_STEPS_REQUIRED:
                    chunk_res = copy.copy(res)
                    chunk_res.data = block_df.copy().reset_index(drop=True)

                    valid_results.append(chunk_res)
                    part_counter += 1

        if part_counter > 2:
            logger.info(
                f"Split pair {res.ticker_x}-{res.ticker_y} into {part_counter - 1} episodes."
            )
        elif part_counter == 1:
            logger.warning(f"Pair rejected {res.ticker_x}-{res.ticker_y}.")

    results = valid_results

    if not results:
        logger.error("Data not found")
        wandb.finish()
        return

    logger.info(f"Successfully loaded {len(results)} environments after filtering")

    if wandb.run is not None:
        data_artifact = wandb.Artifact(name="training_dataset", type="dataset")
        data_artifact.add_dir(training_data_dir)
        wandb.log_artifact(data_artifact)

    vec_env = build_multi_env(
        results=results,
        rl_reward=cfg.rl.reward,
        obs_space_type=cfg.rl.obs_space_type,
        fee_rate=cfg.market.fee_rate,
        seed=seed,
        freeze_std=cfg.rl.freeze_std,
        time_decay_stop=cfg.rl.time_decay_stop,
    )
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    algo_name = cfg.rl_algo.algo_name.value
    algo_class = ALGO_MAP.get(algo_name)

    if algo_class is None:
        raise ValueError(f"RL algorithm not found: {algo_name}")

    model_params = cfg.rl_algo.params.model_dump()

    model = algo_class(
        policy=cfg.rl_algo.policy_type.value,
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

        final_model_name = (
            f"{algo_name}_{cfg.rl.obs_space_type.value}_{run.id}_seed{seed}"
        )
        save_path = f"{model_dir}/{final_model_name}"
        model.save(save_path)
        vec_env.save(f"{save_path}_normalize.pkl")

        checkpoint_dir = f"{model_dir}/{run.id}"
        if os.path.exists(checkpoint_dir):
            shutil.rmtree(checkpoint_dir)
            logger.debug(f"Deleted checkpoint: {checkpoint_dir}")

        if wandb.run is not None:
            model_artifact = wandb.Artifact(
                name=f"{algo_name}_{cfg.rl.obs_space_type.value}_model_{run.id}",
                type="model",
                description=f"Trained {algo_name} model (Space: {cfg.rl.obs_space_type.value})",
            )
            model_artifact.add_file(f"{save_path}.zip")
            model_artifact.add_file(f"{save_path}_normalize.pkl")
            wandb.log_artifact(model_artifact)
            logger.info("Logged model artifact to W&B.")

    except KeyboardInterrupt:
        logger.info("Training interrupted manually. Saving current model...")
        final_model_name = (
            f"{algo_name}_{cfg.rl.obs_space_type.value}_{run.id}_seed{seed}_interrupted"
        )
        save_path = f"{model_dir}/{final_model_name}"
        model.save(save_path)
        vec_env.save(f"{save_path}_normalize.pkl")

        checkpoint_dir = f"{model_dir}/{run.id}"
        if os.path.exists(checkpoint_dir):
            shutil.rmtree(checkpoint_dir)
            logger.debug(f"Deleted checkpoint: {checkpoint_dir}")

        if wandb.run is not None:
            model_artifact = wandb.Artifact(
                name=f"{algo_name}_{cfg.rl.obs_space_type.value}_model_{run.id}_interrupted",
                type="model",
                description=f"Interrupted {algo_name} (Space: {cfg.rl.obs_space_type.value})",
            )
            model_artifact.add_file(f"{save_path}.zip")
            model_artifact.add_file(f"{save_path}_normalize.pkl")
            wandb.log_artifact(model_artifact)
            logger.info("Saved interrupted model.")
    finally:
        wandb.finish()


if __name__ == "__main__":
    train_agent()
