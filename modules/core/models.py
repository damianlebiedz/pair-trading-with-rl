from dataclasses import dataclass
import pandas as pd


@dataclass(frozen=True)
class ExecutionContext:
    ticker_x: str
    ticker_y: str
    fee_rate: float


@dataclass(frozen=True)
class PositionContext:
    base_sl_thr: float


@dataclass
class PositionState:
    position: float = 0
    prev_position: float = 0
    q_x: float = 0
    q_y: float = 0
    w_x: float | None = None
    w_y: float | None = None
    prev_dif: float | None = None
    time_in_pos: int = 0
    sl_thr: float | None = None

    def update_position(
        self,
        position,
        q_x,
        q_y,
        w_x,
        w_y,
        prev_dif,
        sl_thr,
    ):
        self.position = position
        self.q_x = q_x
        self.q_y = q_y
        self.w_x = w_x
        self.w_y = w_y
        self.prev_dif = prev_dif
        self.sl_thr = sl_thr

    def clear_position(self):
        self.position = 0
        self.q_x = 0
        self.q_y = 0
        self.w_x = None
        self.w_y = None
        self.prev_dif = None
        self.time_in_pos = 0


@dataclass
class StrategyResult:
    data: pd.DataFrame
    ticker_x: str
    ticker_y: str
    start: str
    end: str
    interval: str
    fee_rate: float
    stats: pd.DataFrame | None = None
