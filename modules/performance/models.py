from dataclasses import dataclass, field
from typing import Any
import pandas as pd


@dataclass(slots=True)
class PositionState:
    position: float = 0.0
    prev_position: float = 0.0
    q_x: float = 0.0
    q_y: float = 0.0
    w_x: float | None = None
    w_y: float | None = None
    entry_dif: float | None = None
    prev_dif: float | None = None
    entry_equity: float = 0.0
    time_in_pos: int = 0
    sl_thr: float | None = None
    open_time: pd.Timestamp | None = None
    entry_beta: float | None = None
    entry_win: int | None = None
    entry_std: float | None = None
    sl_lock: bool = False

    def update_position(
        self,
        position,
        q_x,
        q_y,
        w_x,
        w_y,
        entry_dif,
        prev_dif,
        entry_equity,
        sl_thr,
        entry_beta,
        entry_win,
        entry_std,
    ):
        self.position = position
        self.q_x = q_x
        self.q_y = q_y
        self.w_x = w_x
        self.w_y = w_y
        self.entry_dif = entry_dif
        self.prev_dif = prev_dif
        self.entry_equity = entry_equity
        self.sl_thr = sl_thr
        self.entry_beta = entry_beta
        self.entry_win = entry_win
        self.entry_std = entry_std

    def clear_position(self):
        self.position = 0.0
        self.q_x = 0.0
        self.q_y = 0.0
        self.w_x = None
        self.w_y = None
        self.entry_dif = None
        self.prev_dif = None
        self.entry_equity = 0.0
        self.time_in_pos = 0
        self.entry_beta = None
        self.entry_win = None
        self.entry_std = None


@dataclass(slots=True)
class ExecLogger:
    _logs: list[dict[str, Any]] = field(default_factory=list)

    def log(
        self,
        open_time: pd.Timestamp,
        price_x: float,
        price_y: float,
        qx: float,
        qy: float,
        position: float,
        fees: float,
        pnl: float,
        entry_equity: float,
        time_in_pos: int,
    ):
        self._logs.append(
            {
                "open_time": open_time,
                "price_x": price_x,
                "price_y": price_y,
                "qx": qx,
                "qy": qy,
                "position": position,
                "fees": fees,
                "pnl": pnl,
                "entry_equity": entry_equity,
                "time_in_pos": time_in_pos,
            }
        )

    def to_df(self) -> pd.DataFrame:
        if not self._logs:
            return pd.DataFrame()
        return pd.DataFrame(self._logs)


@dataclass(slots=True)
class StrategyResult:
    data: pd.DataFrame
    ticker_x: str
    ticker_y: str
    start: str
    end: str
    interval: str
    fee_rate: float
    stats: pd.DataFrame | None = None
    exec_logger: pd.DataFrame | None = None
