from dataclasses import dataclass, fields
import numpy as np

from modules.core.enums import ObsSpaceType


@dataclass(slots=True)
class AgentState:
    """
    Encapsulates the observable state of the trading environment for the RL agent.

    Attributes:
        market_z_score (float): Market Z-Score of the spread between the two assets.
        z_score (float): Z-Score based on entry beta-hedge and std (when in-position), equals to market_z_score when out-of-position.
        market_beta (float): Market hedge-ratio between two assets, used to size positions.
        market_std (float): Market standard deviation of the spread.
        hurst (float): Hurst Exponent value.
        window (int): Lookback window length used for z-score calculation.
        position (float): Current position in the strategy (-1, 0, 1 scaled by capital).
        signal (float): Position (signal) from non-RL backtest.
        norm_time_in_pos (float): Normalized time in current position (0.0–1.0), where 1.0 means equal to window length.
        drawdown_pct (float): Current drawdown as a fraction of peak equity.
        current_market_vol (float): Instantaneous market volatility, computed as average of both assets' rolling volatility.

    Methods:
        get_state_arr() -> np.ndarray:
            Converts the dataclass into a fixed-size float32 array suitable as
            input to RL agents.
    """

    market_z_score: float
    z_score: float
    market_beta: float
    market_std: float
    hurst: float
    window: int
    position: float
    signal: float
    norm_time_in_pos: float
    drawdown_pct: float
    current_market_vol: float

    @classmethod
    def get_obs_shape(cls, obs_space_type: ObsSpaceType) -> tuple[int]:
        """
        Dynamically calculates the shape of the observation array.

        It creates a dummy state instance with zeroed fields to evaluate
        the array length for a given observation space type, preventing
        the need for hardcoded dimensions.
        """
        dummy_kwargs = {f.name: 0 if f.type is int else 0.0 for f in fields(cls)}
        dummy_state = cls(**dummy_kwargs)
        return dummy_state.get_state_arr(obs_space_type).shape

    def get_state_arr(
        self,
        obs_space_type: ObsSpaceType,
    ) -> np.ndarray:
        """
        obs_space_type == "minimal":
            - state_arr: market_z_score, z_score, position, norm_time_in_pos, signal
        obs_space_type == "standard":
            - state_arr: market_z_score, z_score, market_beta, market_std, hurst, position, norm_time_in_pos, signal
        obs_space_type == "full":
            - state_arr: market_z_score, z_score, market_beta, market_std, hurst, window, position, norm_time_in_pos, signal, drawdown_pct, current_market_vol
        """
        if obs_space_type == ObsSpaceType.MINIMAL:
            return np.array(
                [
                    self.market_z_score,
                    self.z_score,
                    self.position,
                    self.norm_time_in_pos,
                    self.signal,
                ],
                dtype=np.float32,
            )

        elif obs_space_type == ObsSpaceType.STANDARD:
            return np.array(
                [
                    self.market_z_score,
                    self.z_score,
                    self.market_beta,
                    self.market_std,
                    self.hurst,
                    self.position,
                    self.norm_time_in_pos,
                    self.signal,
                ],
                dtype=np.float32,
            )

        else:
            return np.array(
                [
                    self.market_z_score,
                    self.z_score,
                    self.market_beta,
                    self.market_std,
                    self.hurst,
                    self.window,
                    self.position,
                    self.norm_time_in_pos,
                    self.signal,
                    self.drawdown_pct,
                    self.current_market_vol,
                ],
                dtype=np.float32,
            )
