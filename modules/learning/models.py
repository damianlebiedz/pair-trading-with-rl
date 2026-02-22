from dataclasses import dataclass
from typing import Literal
import numpy as np


@dataclass(slots=True)
class AgentState:
    """
    Encapsulates the observable state of the trading environment for the RL agent.

    Attributes:
        z_score (float): Current z-score of the spread between the two assets.
        std (float): Rolling standard deviation of the spread.
        beta (float): Hedge ratio between the two assets, used to size positions.
        hurst (float): Hurst Exponent value.
        window (int): Lookback window length used for z-score calculation.
        position (float): Current position in the strategy (-1, 0, 1 scaled by capital).
        norm_time_in_pos (float): Normalized time in current position (0.0–1.0), where 1.0 means equal to window length.
        drawdown_pct (float): Current drawdown as a fraction of peak equity.
        current_market_vol (float): Instantaneous market volatility, computed as average of both assets' rolling volatility.

    Methods:
        get_state_arr() -> np.ndarray:
            Converts the dataclass into a fixed-size float32 array suitable as
            input to RL agents.
    """

    z_score: float
    std: float
    beta: float
    hurst: float
    window: int
    position: float
    norm_time_in_pos: float
    drawdown_pct: float
    current_market_vol: float

    def get_state_arr(
        self,
        obs_space_type: Literal["full", "standard", "minimal"],
    ) -> np.ndarray:
        if obs_space_type == "minimal":
            return np.array(
                [self.z_score, self.position, self.norm_time_in_pos],
                dtype=np.float32,
            )

        elif obs_space_type == "standard":
            return np.array(
                [self.z_score, self.std, self.beta, self.hurst, self.position, self.norm_time_in_pos],
                dtype=np.float32,
            )

        else:
            return np.array(
                [
                    self.z_score,
                    self.std,
                    self.beta,
                    self.hurst,
                    float(self.window),
                    self.position,
                    self.norm_time_in_pos,
                    self.drawdown_pct,
                    self.current_market_vol,
                ],
                dtype=np.float32,
            )
