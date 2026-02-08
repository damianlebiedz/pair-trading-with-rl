import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from modules.core.models import ExecutionContext, PositionState
from modules.core.execution import TradeExecutor


class PairsTradingEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, df: pd.DataFrame, exec_ctx: ExecutionContext):
        super(PairsTradingEnv, self).__init__()

        self.df = df.reset_index(drop=True)
        self.ctx = exec_ctx

        required_cols = ["z_score", "spread", "mean", "std", "beta", "win", self.ctx.ticker_x, self.ctx.ticker_y]
        missing = [c for c in required_cols if c not in self.df.columns]
        if missing:
            raise ValueError(f"Missing columns in df: {missing}.")

        # --- ACTION SPACE (DISCRETE) ---
        # 0 -> SHORT (-1.0)
        # 1 -> FLAT  ( 0.0)
        # 2 -> LONG  ( 1.0)
        self.action_space = spaces.Discrete(3)

        # --- OBSERVATION SPACE ---
        # [Z-Score, Spread, Mean, Std, Beta, Position_State(-1/0/1), Net_PnL_Pct]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32
        )

        self.current_step = 0
        self.position_state = PositionState()
        self.portfolio_value = self.ctx.initial_cash
        self.total_fees = 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.position_state = PositionState()
        self.portfolio_value = self.ctx.initial_cash
        self.total_fees = 0.0
        return self._get_observation(), {}

    def step(self, action):
        mapping = {
            0: -1.0,    # Short
            1: 0.0,     # Flat / Exit
            2: 1.0      # Long
        }
        trade_action = mapping[int(action)]

        price_x = self.df.at[self.current_step, self.ctx.ticker_x]
        price_y = self.df.at[self.current_step, self.ctx.ticker_y]
        z_score = self.df.at[self.current_step, "z_score"]
        beta = self.df.at[self.current_step, "beta"]
        current_net_return_pct = self.df.at[self.current_step, "net_return_pct"]

        step_pnl, step_fees = TradeExecutor.execute(
            ctx=self.ctx,
            position_state=self.position_state,
            action=trade_action,
            price_x=price_x,
            price_y=price_y,
            z_score=z_score,
            beta=beta,
            portfolio_value=self.portfolio_value,
            total_fees=self.total_fees,
            exit_threshold=None
        )

        self.portfolio_value += step_pnl
        self.total_fees = step_fees

        # Reward
        reward = step_pnl

        # Next step
        self.current_step += 1
        terminated = self.current_step >= len(self.df) - 1
        truncated = False

        info = {
            "portfolio_value": self.portfolio_value,
            "position": self.position_state.position,
            "action": trade_action
        }

        return self._get_observation(), reward, terminated, truncated, info

    def _get_observation(self):
        if self.current_step >= len(self.df):
            return np.zeros(self.observation_space.shape, dtype=np.float32)

        row = self.df.iloc[self.current_step]

        obs = np.array([
            row.get("z_score", 0.0),
            row.get("spread", 0.0),
            row.get("mean", 0.0),
            row.get("std", 0.0),
            row.get("beta", 1.0),
            float(self.position_state.position),  # -1.0, 0.0, 1.0
            (self.portfolio_value - self.ctx.initial_cash) / self.ctx.initial_cash
        ], dtype=np.float32)

        return np.nan_to_num(obs)
