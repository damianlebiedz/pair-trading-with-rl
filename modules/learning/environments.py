import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from stable_baselines3.common.monitor import Monitor

from modules.core.enums import ObsSpaceType, RLRewards, Source
from modules.core.indicators import calculate_z_score, calculate_spread_statistics
from modules.performance.models import (
    StrategyResult,
    PositionState,
)
from modules.core.execution import TradeExecutor
from modules.learning.models import AgentState
from modules.learning.rewards import RewardScheme, PnLReward, PnLSignalReward
from stable_baselines3.common.vec_env import DummyVecEnv


def build_multi_env(
    results: list[StrategyResult],
    rl_reward: str,
    obs_space_type: ObsSpaceType,
    fee_rate: float,
    seed: int = None,
) -> DummyVecEnv:
    env_fns = []

    for res in results:

        def make_env(result=res):
            reward_map = {
                RLRewards.PNL: PnLReward,
                RLRewards.PNL_SIGNAL: PnLSignalReward,
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
    def __init__(self, obs_space_type: ObsSpaceType):
        super().__init__()

        obs_shape = AgentState.get_obs_shape(obs_space_type)

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
        obs_space_type: ObsSpaceType,
        fee_rate: float,
    ):
        super(PairsTradingEnv, self).__init__()

        self.result = result
        self.df = result.data.reset_index(drop=True)
        self.position_state = PositionState()
        self.action_space = spaces.Discrete(3)
        self.fee_rate = fee_rate

        obs_shape = AgentState.get_obs_shape(obs_space_type)

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

        self.ep_trades = 0
        self.ep_wins = 0
        self.ep_total_hold_time = 0
        self.ep_total_fees = 0.0

        self.current_trade_net_pnl = 0.0
        self.current_trade_duration = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.current_step = 0
        self.equity = self.initial_equity
        self.peak_equity = self.initial_equity
        self.position_state.clear_position()
        self._update_state_object()
        self.reward_scheme.reset()

        self.ep_trades = 0
        self.ep_wins = 0
        self.ep_total_hold_time = 0
        self.ep_total_fees = 0.0
        self.current_trade_net_pnl = 0.0
        self.current_trade_duration = 0

        window = int(self.df["window"].dropna().iloc[0])
        self.current_step = window - 1

        self._update_state_object()

        return self.state.get_state_arr(self.obs_space_type), {}

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

            exec_win = int(row["window"])
            start_idx = max(0, self.current_step - exec_win + 1)

            source_x_col = f"{self.result.ticker_x}_{Source.LOG.value}"
            source_y_col = f"{self.result.ticker_y}_{Source.LOG.value}"

            slice_x = (
                self.df[source_x_col].iloc[start_idx : self.current_step + 1].values
            )
            slice_y = (
                self.df[source_y_col].iloc[start_idx : self.current_step + 1].values
            )

            _, _, current_std = calculate_spread_statistics(
                X_slice=slice_x, Y_slice=slice_y, beta=exec_beta
            )

            exec_std = self.position_state.entry_std
        else:
            exec_beta = row["market_beta"]
            exec_std = row["market_std"]

        exec_win = row["window"]

        step_pnl, step_fees = TradeExecutor.execute(
            fee_rate=self.result.fee_rate,
            position_state=self.position_state,
            stop_loss_thr=None,
            action=target_position,
            price_x=price_x,
            price_y=price_y,
            beta=exec_beta,
            equity=self.equity,
            exec_logger=None,
            std=exec_std,
            sl_lock=False,
        )

        self.equity += step_pnl - step_fees
        self.peak_equity = max(self.peak_equity, self.equity)

        prev_pos = self.position_state.prev_position
        curr_pos = self.position_state.position

        self.ep_total_fees += step_fees

        if prev_pos != 0 or curr_pos != 0:
            self.current_trade_net_pnl += step_pnl - step_fees

        if prev_pos != 0:
            self.current_trade_duration += 1

        trade_ended = (prev_pos != 0) and (curr_pos != prev_pos)

        if trade_ended:
            self.ep_trades += 1
            if self.current_trade_net_pnl > 0:
                self.ep_wins += 1
            self.ep_total_hold_time += self.current_trade_duration

            self.current_trade_net_pnl = 0.0
            self.current_trade_duration = 0

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
            "ep_total_fees": self.ep_total_fees,
        }

        if self.ep_trades > 0:
            info["win_rate"] = self.ep_wins / self.ep_trades
            info["avg_hold_time"] = self.ep_total_hold_time / self.ep_trades
        else:
            info["win_rate"] = 0.0
            info["avg_hold_time"] = 0.0

        reward = self.reward_scheme.calculate(
            step_pnl=step_pnl,
            equity=self.equity,
            position=self.position_state.position,
            signal=position,
            step_fees=step_fees,
            is_bankrupt=is_bankrupt,
            fee_rate=self.fee_rate,
            win=exec_win,
        )

        return self._get_observation(), reward, terminated, truncated, info

    def _get_observation(self) -> np.ndarray:
        if self.state is None:
            return np.zeros(self.observation_space.shape, dtype=np.float32)

        obs = self.state.get_state_arr(self.obs_space_type)
        return np.array(obs, dtype=np.float32)

    def _update_state_object(self):
        row = self.df.iloc[self.current_step]

        window = row.get("window")
        market_z_score = row.get("market_z_score")
        market_beta = row.get("market_beta")
        market_std = row.get("market_std")
        hurst = row.get("hurst")
        market_vol = row.get("market_vol")
        signal = row.get("position")

        if (
            self.position_state.position != 0
            and self.position_state.entry_beta is not None
        ):
            start_idx = max(0, self.current_step - window + 1)

            source_x_col = f"{self.result.ticker_x}_{Source.LOG.value}"
            source_y_col = f"{self.result.ticker_y}_{Source.LOG.value}"

            slice_x = (
                self.df[source_x_col].iloc[start_idx : self.current_step + 1].values
            )
            slice_y = (
                self.df[source_y_col].iloc[start_idx : self.current_step + 1].values
            )

            spread, mean, _ = calculate_spread_statistics(
                X_slice=slice_x, Y_slice=slice_y, beta=self.position_state.entry_beta
            )

            std = self.position_state.entry_std

            z_score = calculate_z_score(spread=spread, mean=mean, std=std)
        else:
            z_score = market_z_score

        time_in_pos = self.position_state.time_in_pos
        norm_time = 0.0
        if pd.notna(window) and window > 0:
            norm_time = time_in_pos / window

        drawdown_pct = 0.0
        if self.peak_equity > 0:
            drawdown_pct = (self.peak_equity - self.equity) / self.peak_equity

        self.state = AgentState(
            market_z_score=float(market_z_score),
            z_score=float(z_score),
            market_beta=market_beta,
            market_std=float(market_std),
            hurst=float(hurst),
            window=int(window),
            position=float(self.position_state.position),
            signal=float(signal),
            norm_time_in_pos=float(norm_time),
            drawdown_pct=float(drawdown_pct),
            current_market_vol=float(market_vol),
        )
