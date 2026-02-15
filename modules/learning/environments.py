import gymnasium as gym
import numpy as np
from gymnasium import spaces

from modules.performance.models import (
    StrategyResult,
    ExecLogger,
    PositionState,
)
from modules.core.execution import TradeExecutor
from modules.learning.models import AgentState
from modules.learning.rewards import RewardScheme


class PairsTradingEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        result: StrategyResult,
        reward_scheme: RewardScheme,
    ):
        super(PairsTradingEnv, self).__init__()

        self.result = result
        self.df = result.data.reset_index(drop=True)

        valid_indices = self.df.dropna(subset=["z_score", "hurst", "market_vol", "std", "beta"]).index
        self.warmup_offset = valid_indices[0] if not valid_indices.empty else 0

        self.position_state = PositionState()
        self.exec_logger = ExecLogger()

        required_cols = [
            "z_score",
            "spread",
            "mean",
            "std",
            "beta",
            "window",
            self.result.ticker_x,
            self.result.ticker_y,
        ]

        missing = [c for c in required_cols if c not in self.df.columns]
        if missing:
            raise ValueError(f"Missing columns in df: {missing}")

        # --- ACTION SPACE ---
        # 0 -> SHORT (-1.0)
        # 1 -> FLAT  ( 0.0)
        # 2 -> LONG  ( 1.0)
        self.action_space = spaces.Discrete(3)

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32
        )

        self.initial_equity = self.df["equity"].iloc[0]
        self.equity = self.initial_equity
        self.peak_equity = self.initial_equity

        self.state = None
        self.current_step = 0

        self.reward_scheme = reward_scheme

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.current_step = self.warmup_offset
        self.equity = self.initial_equity
        self.peak_equity = self.initial_equity

        self.position_state.clear_position()
        self._update_state_object()

        self.reward_scheme.reset()

        return self._get_observation(), {}

    def step(self, action):
        # Map discrete action to direction: 0->-1.0, 1->0.0, 2->1.0
        mapping = {0: -1.0, 1: 0.0, 2: 1.0}
        target_position = mapping[int(action)]

        price_x = self.df.at[self.current_step, self.result.ticker_x]
        price_y = self.df.at[self.current_step, self.result.ticker_y]
        beta = self.df.at[self.current_step, "beta"]
        std = self.df.at[self.current_step, "std"]

        step_pnl, step_fees = TradeExecutor.execute(
            fee_rate=self.result.fee_rate,
            position_state=self.position_state,
            stop_loss_thr=None,
            action=target_position,
            price_x=price_x,
            price_y=price_y,
            beta=beta,
            equity=self.equity,
            exec_logger=self.exec_logger,
            std=std,
            sl_lock=None,
        )

        self.equity += step_pnl
        self.peak_equity = max(self.peak_equity, self.equity)

        self.current_step += 1
        terminated = self.current_step >= len(self.df) - 1
        truncated = False

        if not terminated:
            self._update_state_object()

        info = {
            "equity": self.equity,
            "position": self.position_state.position,
            "action": target_position,
            "step_fees": step_fees,
        }

        reward = self.reward_scheme.calculate(
            step_pnl=step_pnl,
            equity=self.equity,
            position=self.position_state.position,
            step_fees=step_fees,
            info=info,
        )

        return self._get_observation(), reward, terminated, truncated, info

    def _update_state_object(self):
        """Constructs the AgentState object based on current step data and position state."""
        row = self.df.iloc[self.current_step]

        window = row.get("window")
        time_in_pos = self.position_state.time_in_pos
        norm_time = time_in_pos / window if window and window > 0 else 0.0

        drawdown_pct = 0.0
        if self.peak_equity > 0:
            drawdown_pct = (self.peak_equity - self.equity) / self.peak_equity

        self.state = AgentState(
            z_score=row.get("z_score", None),
            std=row.get("std", None),
            beta=row.get("beta", None),
            hurst=row.get("hurst", None),
            window=int(window),
            signal=int(self.position_state.position),
            position=float(self.position_state.position),
            norm_time_in_pos=float(norm_time),
            drawdown_pct=float(drawdown_pct),
            current_market_vol=row.get("market_vol"),
        )

    def _get_observation(self) -> np.ndarray:
        if self.state is None:
            return np.zeros(self.observation_space.shape, dtype=np.float32)

        obs = self.state.get_state_arr()
        return np.nan_to_num(obs).astype(np.float32)
