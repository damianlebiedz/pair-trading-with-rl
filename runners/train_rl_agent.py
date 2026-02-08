import pandas as pd
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv
import os

from modules.rl.environments import PairsTradingEnv
from modules.core.models import ExecutionContext


def train_a2c_agent():
    TICKER_X = "AVAXUSDT"
    TICKER_Y = "OPUSDT"
    DATA_PATH = "returns_AVAXUSDT_OPUSDT_2024-11-01_2024-12-01.parquet"
    MODEL_DIR = "models/a2c"
    LOG_DIR = "logs"

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    print(f"Loading training data from {DATA_PATH}...")
    try:
        df = pd.read_parquet(DATA_PATH)
    except FileNotFoundError:
        print(f"Parquet file not found")
        return

    print(f"Columns found: {df.columns.tolist()}")

    exec_ctx = ExecutionContext(
        ticker_x=TICKER_X,
        ticker_y=TICKER_Y,
        fee_rate=0.001
    )

    def make_env():
        return PairsTradingEnv(df=df, exec_ctx=exec_ctx)

    vec_env = DummyVecEnv([make_env])

    # MlpPolicy: Multi-Layer Perceptron
    model = A2C(
        "MlpPolicy",
        vec_env,
        verbose=1,
        tensorboard_log=LOG_DIR,
        learning_rate=0.0007,   # default lr for A2C, can be 3e-4 if unstable
        n_steps=5,              # number of steps until update
        gamma=0.99,             # discount factor
        ent_coef=0.01           # entropy (for exploration)
    )

    print("Starting A2C training...")
    try:
        model.learn(total_timesteps=50000)  # number of iterations
        print("Training finished.")
    except KeyboardInterrupt:
        print("Training interrupted manually. Saving current model...")

    # 7. ZAPIS
    save_path = f"{MODEL_DIR}/a2c_pairs_trader"
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
