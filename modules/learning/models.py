from dataclasses import dataclass
import numpy as np


def _f(x: float | int | None):
    return 0.0 if x is None else float(x)


@dataclass(slots=True)
class AgentState:
    """
    Encapsulates the observable state of the trading environment for the RL agent.

    Attributes:
        z_score (float | None): Current z-score of the spread between the two assets.
        std (float | None): Rolling standard deviation of the spread.
        beta (float): Hedge ratio between the two assets, used to size positions.
        hurst (float): Hurst Exponent value.
        window (int | None): Lookback window length used for z-score calculation.
        signal (int): Trading signal at current timestep: -1 (short), 0 (hold), 1 (long).
        position (float): Current position in the strategy (-1, 0, 1 scaled by capital).
        norm_time_in_pos (float): Normalized time in current position (0.0–1.0), where 1.0 means equal to window length.
        drawdown_pct (float): Current drawdown as a fraction of peak equity.
        current_market_vol (float): Instantaneous market volatility, computed as average of both assets' rolling volatility.

    Methods:
        get_state_arr() -> np.ndarray:
            Converts the dataclass into a fixed-size float32 array suitable as
            input to RL agents. Missing values (None) are replaced with 0.0.
    """

    z_score: float | None
    std: float | None
    beta: float
    hurst: float
    window: int | None
    signal: int
    position: float
    norm_time_in_pos: float
    drawdown_pct: float
    current_market_vol: float

    def get_state_arr(self, normalize: bool = True) -> np.ndarray:
        arr = np.array(
            [
                _f(self.z_score),
                _f(self.std) * 10.0 if normalize else _f(self.std),
                float(self.beta),
                float(self.hurst),
                _f(self.window) / 200.0 if normalize else _f(self.window),
                float(self.signal),
                float(self.position),
                float(self.norm_time_in_pos),
                float(self.drawdown_pct),
                (
                    float(self.current_market_vol) * 50.0
                    if normalize
                    else float(self.current_market_vol)
                ),
            ],
            dtype=np.float32,
        )
        return arr
