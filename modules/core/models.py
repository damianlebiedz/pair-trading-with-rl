from dataclasses import dataclass
import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict


class ExecutionContext(BaseModel):
    ticker_x: str
    ticker_y: str
    fee_rate: float


@dataclass(slots=True)
class PositionState:
    position: float = 0
    prev_position: float = 0
    q_x: float = 0
    q_y: float = 0
    w_x: float | None = None
    w_y: float | None = None
    entry_dif: float | None = None
    prev_dif: float | None = None
    time_in_pos: int = 0
    sl_thr: float | None = None
    open_time: pd.Timestamp | None = None

    def update_position(
        self,
        position,
        q_x,
        q_y,
        w_x,
        w_y,
        prev_dif,
        sl_thr,
        entry_dif,
    ):
        self.position = position
        self.q_x = q_x
        self.q_y = q_y
        self.w_x = w_x
        self.w_y = w_y
        self.prev_dif = prev_dif
        self.sl_thr = sl_thr
        self.entry_dif = entry_dif

    def clear_position(self):
        self.position = 0
        self.q_x = 0
        self.q_y = 0
        self.w_x = None
        self.w_y = None
        self.entry_dif = None
        self.prev_dif = None
        self.time_in_pos = 0


@dataclass(slots=True)
class Log:
    open_time: pd.Timestamp
    price_x: float
    price_y: float
    qx: float
    qy: float
    position: float
    fees: float
    pnl: float | None = None
    time_in_pos: int | None = None


class ExecLogger:
    def __init__(self):
        self._buffer = []

    def append(self, log: Log):
        self._buffer.append(log)

    def to_df(self) -> pd.DataFrame:
        return pd.DataFrame(self._buffer)


class StrategyResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    data: pd.DataFrame
    ticker_x: str
    ticker_y: str
    start: str
    end: str
    interval: str
    fee_rate: float
    stats: pd.DataFrame | None = None
    exec_logger: pd.DataFrame | None = None


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
        window (int | None): Lookback window length used for z-score calculation.
        signal (int): Trading signal at current timestep: -1 (short), 0 (hold), 1 (long).
        position (float): Current position in the strategy (-1, 0, 1 scaled by capital).
        norm_time_in_pos (float): Normalized time in current position (0.0–1.0), where 1.0 means equal to window length.
        drawdown_pct (float): Current drawdown as a fraction of peak equity.
        current_market_vol (float): Instantaneous market volatility, computed as average of both assets' rolling volatility.
        sl_utilization (float | None): Fraction of stop-loss distance currently “used”.
            0.0 means just entered, 1.0 means hit stop-loss threshold. None if not applicable.

    Methods:
        get_state_arr() -> np.ndarray:
            Converts the dataclass into a fixed-size float32 array suitable as
            input to RL agents. Missing values (None) are replaced with 0.0.
    """

    z_score: float | None
    std: float | None
    beta: float
    window: int | None
    signal: int
    position: float
    norm_time_in_pos: float
    drawdown_pct: float
    current_market_vol: float
    sl_utilization: float | None

    def get_state_arr(self) -> np.ndarray:
        return np.array(
            [
                _f(self.z_score),
                _f(self.std),
                float(self.beta),
                _f(self.window),
                float(self.signal),
                float(self.position),
                float(self.norm_time_in_pos),
                float(self.drawdown_pct),
                float(self.current_market_vol),
                _f(self.sl_utilization),
            ],
            dtype=np.float32,
        )
