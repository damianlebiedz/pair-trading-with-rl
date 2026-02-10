import logging
from datetime import datetime
from pathlib import Path
import hydra
from omegaconf import DictConfig
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv
import os

from modules.data_services.data_utils import load_strategy_result
from modules.rl.environments import PairsTradingEnv
from modules.core.models import ExecutionContext
from runners.core.pipelines import setup_rl_run_environment

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../conf", config_name="base")
def train_a2c_agent(cfg: DictConfig):
    rl_root = setup_rl_run_environment(__file__)

    data_path = os.path.join(rl_root, "training_data")
    model_dir = os.path.join(rl_root, "models")
    log_dir = os.path.join(rl_root, "tensorboard_logs")

    os.makedirs(data_path, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    logger.info(f"Loading training data from '{data_path}'...")

    try:
        latest_file = max(
            Path(data_path).glob("*.parquet"), key=lambda p: p.stat().st_mtime
        )
        result = load_strategy_result(latest_file.name, directory="training_data")

    except (ValueError, FileNotFoundError) as e:
        logger.error(f"Parquet file not found or error loading in '{data_path}': {e}")
        return

    logger.info(f"Loaded data for {result.ticker_x}/{result.ticker_y}")

    exec_ctx = ExecutionContext(
        ticker_x=result.ticker_x,
        ticker_y=result.ticker_y,
        fee_rate=result.fee_rate
    )

    def make_env():
        return PairsTradingEnv(result=result, exec_ctx=exec_ctx, reward_scheme=cfg.rl_reward)

    vec_env = DummyVecEnv([make_env])

    model = A2C(
        cfg.policy_type,
        vec_env,
        verbose=cfg.verbose,
        tensorboard_log=log_dir,
        learning_rate=cfg.learning_rate,
        n_steps=cfg.n_steps,
        gamma=cfg.gamma,
        ent_coef=cfg.ent_coef,
    )

    logger.info("Starting A2C training...")
    try:
        model.learn(total_timesteps=cfg.gettotal_timesteps)
        logger.info("Training finished.")
    except KeyboardInterrupt:
        logger.info("Training interrupted manually. Saving current model...")

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_path = f"{model_dir}/a2c_{timestamp}"
    model.save(save_path)
    logger.info(f"Model saved to {save_path}.zip")

    logger.info("Running quick validation...")
    obs = vec_env.reset()
    total_reward = 0

    for _ in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = vec_env.step(action)

        total_reward += rewards[0]
        if dones[0]:
            break

    logger.info(f"Validation Total Reward (PnL): {total_reward:.2f}")


if __name__ == "__main__":
    train_a2c_agent()
