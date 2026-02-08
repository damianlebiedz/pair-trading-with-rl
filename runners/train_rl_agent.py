import logging
from datetime import datetime
from pathlib import Path
import hydra
from omegaconf import DictConfig
import pandas as pd
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv
import os

from modules.rl.environments import PairsTradingEnv
from modules.core.models import ExecutionContext
from runners.core.pipelines import setup_rl_run_environment

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../conf", config_name="base")
def train_a2c_agent(cfg: DictConfig):
    rl_root = setup_rl_run_environment(__file__)

    TICKER_X = "AVAXUSDT"
    TICKER_Y = "OPUSDT"

    data_path = os.path.join(rl_root, "training_data")
    model_dir = os.path.join(rl_root, "models")
    log_dir = os.path.join(rl_root, "tensorboard_logs")

    os.makedirs(data_path, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    print(f"Loading training data from '{data_path}'...")

    try:
        latest_file = max(
            Path(data_path).glob("*.parquet"), key=lambda p: p.stat().st_mtime
        )
        df = pd.read_parquet(latest_file)

    except ValueError:
        print(f"Parquet file not found in '{data_path}'")
        return

    print(f"Columns found: {df.columns.tolist()}")

    exec_ctx = ExecutionContext(ticker_x=TICKER_X, ticker_y=TICKER_Y, fee_rate=0.001)

    def make_env():
        return PairsTradingEnv(df=df, exec_ctx=exec_ctx)

    vec_env = DummyVecEnv([make_env])

    # MlpPolicy: Multi-Layer Perceptron
    model = A2C(
        cfg.policy_type,
        vec_env,
        verbose=1,
        tensorboard_log=log_dir,
        learning_rate=cfg.learning_rate,  # default lr for A2C, can be 3e-4 if unstable
        n_steps=cfg.n_steps,  # number of steps until update
        gamma=cfg.gamma,  # discount factor
        ent_coef=cfg.ent_coef,  # entropy (for exploration)
    )

    print("Starting A2C training...")
    try:
        model.learn(total_timesteps=cfg.total_timestamps)  # number of iterations
        print("Training finished.")
    except KeyboardInterrupt:
        print("Training interrupted manually. Saving current model...")

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_path = f"{model_dir}/a2c_{timestamp}"
    model.save(save_path)
    print(f"Model saved to {save_path}.zip")

    print("Running quick validation...")
    obs = vec_env.reset()
    total_reward = 0

    for _ in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = vec_env.step(action)
        total_reward += rewards[0]
        if dones[0]:
            break

    print(f"Validation Total Reward (PnL): {total_reward:.2f}")


if __name__ == "__main__":
    train_a2c_agent()
