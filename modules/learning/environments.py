from typing import Literal
import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from stable_baselines3.common.monitor import Monitor

from modules.performance.models import (
    StrategyResult,
    PositionState,
)
from modules.core.execution import TradeExecutor
from modules.learning.models import AgentState
from modules.learning.rewards import RewardScheme, PnLSignalReward
from stable_baselines3.common.vec_env import DummyVecEnv
from modules.learning.rewards import (
    PnLReward,
    DifferentialSharpeReward,
)


def build_multi_env(
    results: list[StrategyResult],
    rl_reward: str,
    obs_space_type: Literal["full", "standard", "minimal"],
    fee_rate: float,
    seed: int = None,
) -> DummyVecEnv:
    env_fns = []

    for res in results:

        def make_env(result=res):
            reward_map = {
                "pnl": PnLReward,
                "pnl_signal": PnLSignalReward,
                "diff_sharpe": DifferentialSharpeReward,
            }
            reward_schema = reward_map[rl_reward]()
            env = PairsTradingEnv(
                result=result,
                reward_scheme=reward_schema,
                obs_space_type=obs_space_type,
                fee_rate=fee_rate,
            )
            return Monitor(env)

        env_fns.append(make_env)

    vec_env = DummyVecEnv(env_fns)
    if seed is not None:
        vec_env.seed(seed)

    return vec_env


class MockEnv(gym.Env):
    def __init__(self, obs_space_type: Literal["full", "standard", "minimal"]):
        super().__init__()

        if obs_space_type == "minimal":
            obs_shape = (4,)
        elif obs_space_type == "standard":
            obs_shape = (7,)
        else:
            obs_shape = (10,)

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=obs_shape, dtype=np.float32
        )
        self.action_space = spaces.Discrete(3)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        return obs, {}

    def step(self, action):
        obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        return obs, 0.0, False, False, {}


class PairsTradingEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        result: StrategyResult,
        reward_scheme: RewardScheme,
        obs_space_type: Literal["full", "standard", "minimal"],
        fee_rate: float,
    ):
        super(PairsTradingEnv, self).__init__()

        self.result = result
        self.df = result.data.reset_index(drop=True)
        self.position_state = PositionState()
        self.action_space = spaces.Discrete(3)
        self.fee_rate = fee_rate

        if obs_space_type == "minimal":
            obs_shape = (4,)
        elif obs_space_type == "standard":
            obs_shape = (7,)
        else:
            obs_shape = (10,)

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=obs_shape, dtype=np.float32
        )

        self.initial_equity = self.df["equity"].iloc[0]
        self.equity = self.initial_equity
        self.peak_equity = self.initial_equity

        self.state = None
        self.current_step = 0

        self.reward_scheme = reward_scheme
        self.obs_space_type = obs_space_type

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.current_step = 0
        self.equity = self.initial_equity
        self.peak_equity = self.initial_equity
        self.position_state.clear_position()
        self._update_state_object()
        self.reward_scheme.reset()

        return self._get_observation(), {}

    def step(self, action):
        self.position_state.prev_position = self.position_state.position

        mapping = {0: -1.0, 1: 0.0, 2: 1.0}
        target_position = mapping[int(action)]

        row = self.df.iloc[self.current_step]
        price_x = row[self.result.ticker_x]
        price_y = row[self.result.ticker_y]
        position = row["position"]

        if (
            self.position_state.position != 0
            and self.position_state.entry_beta is not None
        ):
            exec_beta = self.position_state.entry_beta
            exec_std = self.position_state.entry_std
            exec_win = self.position_state.entry_win
        else:
            exec_beta = row["market_beta"]
            exec_std = row["market_std"]
            exec_win = row["market_win"]

        step_pnl, step_fees = TradeExecutor.execute(
            fee_rate=self.result.fee_rate,
            position_state=self.position_state,
            stop_loss_thr=None,
            action=target_position,
            price_x=price_x,
            price_y=price_y,
            beta=exec_beta,
            win=exec_win,
            equity=self.equity,
            exec_logger=None,
            std=exec_std,
            sl_lock=False,
        )

        self.equity += step_pnl - step_fees
        self.peak_equity = max(self.peak_equity, self.equity)

        self.current_step += 1

        is_bankrupt = self.equity <= 0.0
        terminated = (self.current_step >= len(self.df) - 1) or is_bankrupt
        truncated = False

        if not terminated:
            self._update_state_object()
        elif is_bankrupt:
            step_pnl = -self.initial_equity
            self.equity = 0.0
            self.position_state.clear_position()

        info = {
            "equity": self.equity,
            "position": self.position_state.position,
            "action": target_position,
            "step_fees": step_fees,
            "is_bankrupt": is_bankrupt,
        }

        reward = self.reward_scheme.calculate(
            step_pnl=step_pnl,
            equity=self.equity,
            position=self.position_state.position,
            signal=position,
            step_fees=step_fees,
            is_bankrupt=is_bankrupt,
            fee_rate=self.fee_rate,
            market_win=exec_win,
        )

        return self._get_observation(), reward, terminated, truncated, info

    def _get_observation(self) -> np.ndarray:
        if self.state is None:
            return np.zeros(self.observation_space.shape, dtype=np.float32)

        obs = self.state.get_state_arr(self.obs_space_type)
        return np.array(obs, dtype=np.float32)

    def _update_state_object(self):
        row = self.df.iloc[self.current_step]

        market_win = row.get("market_win")
        market_z_score = row.get("market_z_score")
        market_std = row.get("market_std")
        market_beta = row.get("market_beta")
        market_hurst = row.get("hurst")
        market_vol = row.get("market_vol")
        position = row.get("position")

        time_in_pos = self.position_state.time_in_pos
        norm_time = 0.0
        if pd.notna(market_win) and market_win > 0:
            norm_time = time_in_pos / market_win

        drawdown_pct = 0.0
        if self.peak_equity > 0:
            drawdown_pct = (self.peak_equity - self.equity) / self.peak_equity

        self.state = AgentState(
            z_score=float(market_z_score),
            std=float(market_std),
            beta=float(market_beta),
            hurst=float(market_hurst),
            window=int(market_win),
            position=float(self.position_state.position),
            signal=float(position),
            norm_time_in_pos=float(norm_time),
            drawdown_pct=float(drawdown_pct),
            current_market_vol=float(market_vol),
        )
