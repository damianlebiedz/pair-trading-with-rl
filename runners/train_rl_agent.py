import logging
from pathlib import Path
import hydra
import wandb
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf
from stable_baselines3 import A2C
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv
import os
from wandb.integration.sb3 import WandbCallback

from modules.data_services.data_utils import load_strategy_result
from modules.rl.environments import PairsTradingEnv
from modules.core.models import ExecutionContext
from runners.core.pipelines import setup_rl_run_environment

logger = logging.getLogger(__name__)
load_dotenv()


class LogEquityCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(LogEquityCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [{}])

        if infos:
            info = infos[0]

            equity = info.get("equity")
            position = info.get("position")

            if equity is not None and wandb.run is not None:
                wandb.log(
                    {
                        "env/equity": equity,
                        "env/position": position,
                        "global_step": self.num_timesteps,
                    }
                )

        return True


@hydra.main(version_base=None, config_path="../conf", config_name="base")
def train_a2c_agent(cfg: DictConfig):
    rl_root = setup_rl_run_environment(__file__)

    seed = cfg.seed
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
    try:
        latest_file_path = max(
            Path(data_path).glob("*.parquet"), key=lambda p: p.stat().st_mtime
        )
        result = load_strategy_result(latest_file_path.name, directory="training_data")

        if wandb.run is not None:
            data_artifact = wandb.Artifact(
                name="training_dataset",
                type="dataset",
                description=f"Data for {result.ticker_x}/{result.ticker_y} ({result.interval})",
                metadata={"filename": latest_file_path.name},
            )
            data_artifact.add_file(str(latest_file_path))
            wandb.log_artifact(data_artifact)
            logger.info(f"Logged dataset artifact: {latest_file_path.name}")

    except (ValueError, FileNotFoundError) as e:
        logger.error(f"Error loading data: {e}")
        wandb.finish()
        return

    logger.info(f"Loaded data for {result.ticker_x}/{result.ticker_y}")

    exec_ctx = ExecutionContext(
        ticker_x=result.ticker_x, ticker_y=result.ticker_y, fee_rate=result.fee_rate
    )

    def make_env():
        env = PairsTradingEnv(
            result=result, exec_ctx=exec_ctx, reward_scheme=cfg.rl_reward
        )
        env.reset(seed=seed)
        return env

    vec_env = DummyVecEnv([make_env])
    vec_env.seed(seed)

    model = A2C(
        cfg.policy_type,
        vec_env,
        verbose=cfg.verbose,
        tensorboard_log=log_dir,
        learning_rate=cfg.learning_rate,
        n_steps=cfg.n_steps,
        gamma=cfg.gamma,
        ent_coef=cfg.ent_coef,
        seed=seed,
    )

    logger.info(f"Starting A2C training (Run ID: {run.id})...")

    callbacks = [
        WandbCallback(
            gradient_save_freq=100,
            model_save_path=f"{model_dir}/{run.id}",
            verbose=2,
        ),
        LogEquityCallback(),
    ]

    try:
        model.learn(total_timesteps=cfg.total_timesteps, callback=callbacks)
        logger.info("Training finished.")

        final_model_name = f"a2c_{run.id}_seed{seed}"
        save_path = f"{model_dir}/{final_model_name}"
        model.save(save_path)

        if wandb.run is not None:
            model_artifact = wandb.Artifact(
                name=f"a2c_model_{run.id}",
                type="model",
                description="Trained A2C model",
                metadata={"seed": seed, "steps": cfg.total_timesteps},
            )
            model_artifact.add_file(f"{save_path}.zip")
            wandb.log_artifact(model_artifact)
            logger.info("Logged model artifact to W&B.")

            obs = vec_env.reset()
            total_reward = 0

            while True:
                action, _ = model.predict(obs, deterministic=True)
                obs, rewards, dones, info = vec_env.step(action)
                total_reward += rewards[0]
                if dones[0]:
                    break

            logger.info(f"Validation PnL: {total_reward:.2f}")
            wandb.log({"validation/final_pnl": total_reward})

    except KeyboardInterrupt:
        logger.info("Training interrupted manually. Saving current model...")
        final_model_name = f"a2c_{run.id}_seed{seed}_interrupted"
        save_path = f"{model_dir}/{final_model_name}"
        model.save(save_path)

        if wandb.run is not None:
            model_artifact = wandb.Artifact(
                name=f"a2c_model_{run.id}_interrupted", type="model"
            )
            model_artifact.add_file(f"{save_path}.zip")
            wandb.log_artifact(model_artifact)

    finally:
        wandb.finish()


if __name__ == "__main__":
    train_a2c_agent()
