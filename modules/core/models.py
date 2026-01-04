from dataclasses import dataclass
import pandas as pd


@dataclass(frozen=True)
class ExecutionContext:
    ticker_x: str
    ticker_y: str
    initial_cash: float
    fee_rate: float


@dataclass
class PositionState:
    position: float = 0
    prev_position: float = 0
    q_x: float = 0
    q_y: float = 0
    w_x: float | None = None
    w_y: float | None = None
    stop_loss_threshold: float | None = None
    entry_dif: float | None = None

    def update_position(
        self,
        position,
        prev_position,
        q_x,
        q_y,
        w_x,
        w_y,
        stop_loss_threshold,
        entry_dif,
    ):
        self.position = position
        self.prev_position = prev_position
        self.q_x = q_x
        self.q_y = q_y
        self.w_x = w_x
        self.w_y = w_y
        self.stop_loss_threshold = stop_loss_threshold
        self.entry_dif = entry_dif

    def clear_position(self):
        self.position = 0
        self.q_x = 0
        self.q_y = 0
        self.w_x = None
        self.w_y = None
        self.stop_loss_threshold = None
        self.entry_dif = None


@dataclass
class StrategyResult:
    data: pd.DataFrame
    ticker_x: str
    ticker_y: str
    start: str
    end: str
    interval: str
    fee_rate: float
    rolling_window: int
    stats: pd.DataFrame
