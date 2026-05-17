"""Full coverage tests for modules.core.execution."""

from __future__ import annotations

import pandas as pd
import pytest

from modules.core.execution import TradeExecutor
from modules.performance.models import ExecLogger, PositionState


@pytest.fixture
def open_time() -> pd.Timestamp:
    return pd.Timestamp("2024-01-01")


def _sync_prev_position(state: PositionState) -> None:
    state.prev_position = state.position


def _long_position(
    *,
    entry_equity: float = 1000.0,
    entry_dif: float = 0.0,
    prev_dif: float | None = None,
    sl_thr: float | None = None,
) -> PositionState:
    if prev_dif is None:
        prev_dif = entry_dif
    return PositionState(
        position=1.0,
        prev_position=1.0,
        q_x=10.0,
        q_y=-20.0,
        w_x=0.5,
        w_y=0.5,
        entry_dif=entry_dif,
        prev_dif=prev_dif,
        entry_equity=entry_equity,
        sl_thr=sl_thr,
    )


def _short_position(
    *,
    entry_equity: float = 1000.0,
    entry_dif: float = 0.0,
    sl_thr: float | None = None,
) -> PositionState:
    state = PositionState(
        position=-1.0,
        prev_position=-1.0,
        q_x=-10.0,
        q_y=20.0,
        w_x=0.5,
        w_y=0.5,
        entry_dif=entry_dif,
        prev_dif=entry_dif,
        entry_equity=entry_equity,
        sl_thr=sl_thr,
    )
    return state


class TestDecide:
    def test_z_score_none_returns_flat(self) -> None:
        state = PositionState(prev_position=1.0, sl_thr=2.0)
        assert TradeExecutor.decide(state, 1.0, None, 0.5) == (0.0, False, False)

    def test_flat_position_returns_signal(self) -> None:
        state = PositionState(prev_position=0.0)
        assert TradeExecutor.decide(state, -0.5, 1.0, 0.5) == (-0.5, False, False)

    def test_long_reversal_on_opposite_signal(self) -> None:
        state = PositionState(prev_position=1.0, sl_thr=2.0)
        assert TradeExecutor.decide(state, -1.0, 0.0, 0.5) == (-1.0, False, False)

    def test_short_reversal_on_opposite_signal(self) -> None:
        state = PositionState(prev_position=-1.0, sl_thr=2.0)
        assert TradeExecutor.decide(state, 1.0, 0.0, 0.5) == (1.0, False, False)

    def test_long_take_profit(self) -> None:
        state = PositionState(prev_position=1.0, sl_thr=None)
        assert TradeExecutor.decide(state, 1.0, 0.0, 0.5) == (0.0, False, True)

    def test_long_stop_loss(self) -> None:
        state = PositionState(prev_position=1.0, sl_thr=1.0)
        assert TradeExecutor.decide(state, 1.0, -2.0, 0.5) == (0.0, True, False)

    def test_long_take_profit_and_stop_loss(self) -> None:
        state = PositionState(prev_position=1.0, sl_thr=0.5)
        assert TradeExecutor.decide(state, 1.0, -0.5, 0.5) == (0.0, True, True)

    def test_long_hold(self) -> None:
        state = PositionState(prev_position=1.0, sl_thr=5.0)
        assert TradeExecutor.decide(state, 1.0, -1.0, 0.5) == (1.0, False, False)

    def test_short_take_profit(self) -> None:
        state = PositionState(prev_position=-1.0, sl_thr=None)
        assert TradeExecutor.decide(state, -1.0, 0.0, 0.5) == (0.0, False, True)

    def test_short_stop_loss(self) -> None:
        state = PositionState(prev_position=-1.0, sl_thr=1.0)
        assert TradeExecutor.decide(state, -1.0, 2.0, 0.5) == (0.0, True, False)

    def test_short_hold(self) -> None:
        state = PositionState(prev_position=-1.0, sl_thr=5.0)
        assert TradeExecutor.decide(state, -1.0, 2.0, 0.5) == (-1.0, False, False)


class TestExecute:
    def test_flat_no_action(self) -> None:
        state = PositionState()
        pnl, fees = TradeExecutor.execute(
            fee_rate=0.001,
            position_state=state,
            stop_loss_thr=None,
            action=0.0,
            price_x=100.0,
            price_y=50.0,
            beta=1.0,
            equity=10_000.0,
            leverage=2.0,
            exec_logger=None,
            std=0.01,
        )
        assert pnl == 0.0
        assert fees == 0.0

    def test_open_long_from_flat(self) -> None:
        state = PositionState()
        pnl, fees = TradeExecutor.execute(
            fee_rate=0.001,
            position_state=state,
            stop_loss_thr=2.0,
            action=1.0,
            price_x=100.0,
            price_y=50.0,
            beta=1.0,
            equity=10_000.0,
            leverage=2.0,
            exec_logger=None,
            std=0.01,
        )
        assert pnl == 0.0
        assert fees > 0.0
        assert state.position == 1.0
        assert state.q_x > 0
        assert state.q_y < 0
        assert state.sl_thr == 2.0

    def test_open_short_from_flat(self) -> None:
        state = PositionState()
        pnl, fees = TradeExecutor.execute(
            fee_rate=0.001,
            position_state=state,
            stop_loss_thr=None,
            action=-1.0,
            price_x=100.0,
            price_y=50.0,
            beta=2.0,
            equity=5_000.0,
            leverage=1.5,
            exec_logger=None,
            std=0.02,
        )
        assert pnl == 0.0
        assert fees > 0.0
        assert state.position == -1.0
        assert state.q_x < 0
        assert state.q_y > 0

    def test_hold_existing_position(self) -> None:
        state = _long_position()
        pnl, fees = TradeExecutor.execute(
            fee_rate=0.001,
            position_state=state,
            stop_loss_thr=None,
            action=1.0,
            price_x=105.0,
            price_y=50.0,
            beta=1.0,
            equity=10_000.0,
            leverage=2.0,
            exec_logger=None,
            std=0.01,
        )
        assert fees == 0.0
        assert state.time_in_pos == 1
        assert state.position == 1.0
        assert pnl == pytest.approx(50.0)

    def test_close_position(self) -> None:
        state = _long_position(entry_dif=0.0)
        pnl, fees = TradeExecutor.execute(
            fee_rate=0.001,
            position_state=state,
            stop_loss_thr=None,
            action=0.0,
            price_x=110.0,
            price_y=55.0,
            beta=1.0,
            equity=10_000.0,
            leverage=2.0,
            exec_logger=None,
            std=0.01,
        )
        assert fees > 0.0
        assert state.position == 0.0
        assert state.q_x == 0.0

    def test_reversal_long_to_short(self) -> None:
        state = _long_position(entry_dif=0.0)
        pnl, fees = TradeExecutor.execute(
            fee_rate=0.001,
            position_state=state,
            stop_loss_thr=1.5,
            action=-1.0,
            price_x=100.0,
            price_y=50.0,
            beta=1.0,
            equity=10_000.0,
            leverage=2.0,
            exec_logger=None,
            std=0.01,
        )
        assert fees > 0.0
        assert state.position == -1.0
        assert state.q_x < 0

    def test_sl_lock_skips_open_after_close(self) -> None:
        state = _long_position(entry_dif=0.0)
        state.sl_lock = True
        pnl, fees = TradeExecutor.execute(
            fee_rate=0.001,
            position_state=state,
            stop_loss_thr=None,
            action=-1.0,
            price_x=100.0,
            price_y=50.0,
            beta=1.0,
            equity=10_000.0,
            leverage=2.0,
            exec_logger=None,
            std=0.01,
        )
        assert state.position == 0.0
        assert state.q_x == 0.0

    def test_margin_call_forces_close(self) -> None:
        state = _long_position(entry_equity=100.0)
        pnl, fees = TradeExecutor.execute(
            fee_rate=0.001,
            position_state=state,
            stop_loss_thr=None,
            action=1.0,
            price_x=10.0,
            price_y=50.0,
            beta=1.0,
            equity=1000.0,
            leverage=2.0,
            exec_logger=None,
            std=0.01,
        )
        assert state.position == 0.0
        assert pnl < 0.0

    def test_execute_with_exec_logger_on_open_and_close(
        self, open_time: pd.Timestamp
    ) -> None:
        logger = ExecLogger()
        state = PositionState(open_time=open_time)

        TradeExecutor.execute(
            fee_rate=0.001,
            position_state=state,
            stop_loss_thr=None,
            action=1.0,
            price_x=100.0,
            price_y=50.0,
            beta=1.0,
            equity=10_000.0,
            leverage=2.0,
            exec_logger=logger,
            std=0.01,
        )
        assert len(logger._logs) == 1
        assert logger._logs[0]["pnl"] == 0.0
        _sync_prev_position(state)

        TradeExecutor.execute(
            fee_rate=0.001,
            position_state=state,
            stop_loss_thr=None,
            action=0.0,
            price_x=105.0,
            price_y=52.0,
            beta=1.0,
            equity=10_000.0,
            leverage=2.0,
            exec_logger=logger,
            std=0.01,
        )
        assert len(logger._logs) == 2
        assert logger._logs[1]["position"] == 0.0


class TestOpenPosition:
    def test_open_long_direct(self) -> None:
        state = PositionState()
        pnl, fees = TradeExecutor._open_position(
            fee_rate=0.002,
            stop_loss_thr=3.0,
            action=1.0,
            beta=1.0,
            position_state=state,
            price_x=200.0,
            price_y=100.0,
            equity=8000.0,
            exec_logger=None,
            std=0.05,
            leverage=3.0,
        )
        assert pnl == 0.0
        assert fees == pytest.approx(8000.0 * 3.0 * 0.002)
        assert state.w_x == pytest.approx(0.5)
        assert state.w_y == pytest.approx(0.5)

    def test_open_short_direct(self) -> None:
        state = PositionState()
        pnl, fees = TradeExecutor._open_position(
            fee_rate=0.001,
            stop_loss_thr=None,
            action=-1.0,
            beta=3.0,
            position_state=state,
            price_x=50.0,
            price_y=25.0,
            equity=2000.0,
            exec_logger=None,
            std=0.01,
            leverage=1.0,
        )
        assert pnl == 0.0
        assert fees > 0.0
        expected_wx = 1 / 4
        expected_wy = 3 / 4
        assert state.w_x == pytest.approx(expected_wx)
        assert state.w_y == pytest.approx(expected_wy)

    def test_open_with_exec_logger(self, open_time: pd.Timestamp) -> None:
        logger = ExecLogger()
        state = PositionState(open_time=open_time)
        TradeExecutor._open_position(
            fee_rate=0.001,
            stop_loss_thr=None,
            action=1.0,
            beta=1.0,
            position_state=state,
            price_x=100.0,
            price_y=50.0,
            equity=1000.0,
            exec_logger=logger,
            std=0.01,
            leverage=2.0,
        )
        assert len(logger._logs) == 1
        assert logger._logs[0]["open_time"] == open_time


class TestClosePosition:
    def test_close_normal(self) -> None:
        state = _long_position(entry_dif=0.0)
        state.prev_dif = 0.0
        pnl, fees = TradeExecutor._close_position(
            fee_rate=0.001,
            position_state=state,
            price_x=110.0,
            price_y=55.0,
            exec_logger=None,
        )
        assert fees > 0.0
        assert state.position == 0.0

    def test_close_forced_liquidation(self) -> None:
        state = _long_position(entry_dif=0.0, entry_equity=100.0)
        state.prev_dif = 0.0
        pnl, fees = TradeExecutor._close_position(
            fee_rate=0.001,
            position_state=state,
            price_x=10.0,
            price_y=50.0,
            exec_logger=None,
        )
        assert fees == 0.0
        assert pnl == pytest.approx(-100.0)
        assert state.position == 0.0

    def test_close_with_exec_logger(self, open_time: pd.Timestamp) -> None:
        logger = ExecLogger()
        state = _short_position(entry_dif=100.0)
        state.open_time = open_time
        state.prev_dif = 100.0
        state.time_in_pos = 2

        TradeExecutor._close_position(
            fee_rate=0.001,
            position_state=state,
            price_x=95.0,
            price_y=48.0,
            exec_logger=logger,
        )
        assert len(logger._logs) == 1
        assert logger._logs[0]["time_in_pos"] == 3
        assert logger._logs[0]["qx"] == 0.0


class TestHoldPosition:
    def test_hold_updates_state(self) -> None:
        state = _long_position(entry_dif=0.0)
        state.prev_dif = 0.0
        pnl, fees = TradeExecutor._hold_position(
            position_state=state,
            price_x=105.0,
            price_y=52.5,
        )
        assert fees == 0.0
        assert pnl == 0.0
        assert state.time_in_pos == 1
        assert state.prev_dif == pytest.approx(0.0)
        assert state.position == 1.0

    def test_hold_with_nonzero_pnl(self) -> None:
        state = _long_position()
        pnl, fees = TradeExecutor._hold_position(
            position_state=state,
            price_x=105.0,
            price_y=50.0,
        )
        assert fees == 0.0
        assert pnl == pytest.approx(50.0)
        assert state.prev_dif == pytest.approx(50.0)


class TestCallClosePosition:
    def test_delegates_to_close(self) -> None:
        state = _long_position(entry_dif=0.0)
        state.prev_dif = 0.0
        pnl, fees = TradeExecutor.call_close_position(
            fee_rate=0.001,
            position_state=state,
            price_x=100.0,
            price_y=50.0,
            exec_logger=None,
        )
        assert state.position == 0.0
        assert isinstance(pnl, float)
        assert isinstance(fees, float)
