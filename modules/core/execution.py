from modules.performance.models import (
    PositionState,
    ExecLogger,
)


class TradeExecutor:
    """Stateless class for life-cycle management of positions."""

    @staticmethod
    def decide(
        position_state: PositionState,
        signal: float,
        z_score: float | None,
        exit_threshold: float | None,
    ) -> tuple[float, bool]:
        """
        Determines the target position based on strategy rules (Z-score, TP, SL).
        Used by heuristic strategies. RL agents skip this and provide 'action' directly.
        """
        prev_position = position_state.prev_position
        stop_loss_thr = position_state.sl_thr

        if z_score is None:
            return 0.0, False

        if exit_threshold is None:
            return signal, False

        if prev_position != 0:
            is_long = prev_position > 0

            if (is_long and signal < 0) or (not is_long and signal > 0):
                return signal, False

            if is_long:
                hit_tp = z_score >= -exit_threshold
                hit_sl = stop_loss_thr is not None and z_score <= -stop_loss_thr
                if hit_tp or hit_sl:
                    return 0.0, hit_sl
            else:
                hit_tp = z_score <= exit_threshold
                hit_sl = stop_loss_thr is not None and z_score >= stop_loss_thr
                if hit_tp or hit_sl:
                    return 0.0, hit_sl

            return prev_position, False

        else:
            return signal, False

    @classmethod
    def execute(
        cls,
        fee_rate: float,
        position_state: PositionState,
        stop_loss_thr: float | None,
        action: float,
        price_x: float,
        price_y: float,
        beta: float,
        win: int | None,
        equity: float,
        exec_logger: ExecLogger | None,
        std: float | None,
        sl_lock: bool,
    ) -> tuple[float, float]:
        """
        Mechanically aligns the portfolio with the target 'action'.
        Handles Open, Close, Hold, and Reversal logic.
        """
        prev_position = position_state.prev_position

        if action == prev_position:
            if prev_position != 0:
                return cls._hold_position(
                    position_state=position_state, price_x=price_x, price_y=price_y
                )
            return 0.0, 0.0

        pnl_total = 0.0
        fees_total = 0.0
        current_equity = equity

        if prev_position != 0:
            pnl, fees = cls._close_position(
                fee_rate=fee_rate,
                position_state=position_state,
                price_x=price_x,
                price_y=price_y,
                exec_logger=exec_logger,
            )
            pnl_total += pnl
            fees_total += fees
            current_equity = equity + pnl - fees

        if action != 0 and not sl_lock:
            _, fees = cls._open_position(
                fee_rate=fee_rate,
                stop_loss_thr=stop_loss_thr,
                action=action,
                beta=beta,
                win=win,
                position_state=position_state,
                price_x=price_x,
                price_y=price_y,
                equity=current_equity,
                exec_logger=exec_logger,
                std=std,
            )
            fees_total += fees

        return pnl_total, fees_total

    @classmethod
    def _open_position(
        cls,
        fee_rate: float,
        stop_loss_thr: float | None,
        action: float,
        beta: float,
        win: int,
        position_state: PositionState,
        price_x: float,
        price_y: float,
        equity: float,
        exec_logger: ExecLogger | None,
        std: float,
    ) -> tuple[float, float]:
        wx = 1 / (beta + 1)
        wy = beta / (beta + 1)

        pos_cash = equity * abs(action)

        if action > 0:
            qx = pos_cash * wx / price_x
            qy = -(pos_cash * wy) / price_y
        else:
            qx = -(pos_cash * wx) / price_x
            qy = pos_cash * wy / price_y

        entry_dif = qx * price_x + qy * price_y

        position_state.update_position(
            position=action,
            q_x=qx,
            q_y=qy,
            w_x=wx,
            w_y=wy,
            entry_dif=entry_dif,
            prev_dif=entry_dif,
            entry_equity=pos_cash,
            sl_thr=stop_loss_thr,
            entry_beta=beta,
            entry_win=win,
            entry_std=std,
        )

        fees = pos_cash * fee_rate

        if exec_logger:
            exec_logger.log(
                open_time=position_state.open_time,
                price_x=price_x,
                price_y=price_y,
                qx=qx,
                qy=qy,
                position=position_state.position,
                fees=fees,
                pnl=0.0,
                entry_equity=pos_cash,
                time_in_pos=0,
            )

        return 0.0, fees

    @classmethod
    def _close_position(
        cls,
        fee_rate: float,
        position_state: PositionState,
        price_x: float,
        price_y: float,
        exec_logger: ExecLogger | None,
    ) -> tuple[float, float]:
        exit_dif = position_state.q_x * price_x + position_state.q_y * price_y
        exit_val = abs(position_state.q_x) * price_x + abs(position_state.q_y) * price_y

        pnl = exit_dif - position_state.prev_dif
        fees = exit_val * fee_rate

        position_state.time_in_pos += 1

        if exec_logger:
            exec_logger.log(
                open_time=position_state.open_time,
                price_x=price_x,
                price_y=price_y,
                qx=0.0,
                qy=0.0,
                position=0.0,
                fees=fees,
                pnl=exit_dif - position_state.entry_dif,
                entry_equity=position_state.entry_equity,
                time_in_pos=position_state.time_in_pos,
            )

        position_state.clear_position()

        return pnl, fees

    @staticmethod
    def _hold_position(
        position_state: PositionState, price_x: float, price_y: float
    ) -> tuple[float, float]:
        curr_dif = position_state.q_x * price_x + position_state.q_y * price_y

        position_state.position = position_state.prev_position
        position_state.time_in_pos += 1

        pnl = curr_dif - position_state.prev_dif
        position_state.prev_dif = curr_dif

        return pnl, 0.0

    @classmethod
    def call_close_position(
        cls,
        fee_rate: float,
        position_state: PositionState,
        price_x: float,
        price_y: float,
        exec_logger: ExecLogger | None,
    ) -> tuple[float, float]:
        """
        Public wrapper to close a position.

        Purpose:
            Provides a safe, explicit way to close a position at the end of a simulation or
            in special scenarios, without exposing the private `_close_position` logic
            directly. Useful for finalizing results.

        Rationale:
            _close_position is private because normal execution flow should control
            position closing. This wrapper allows intentional closure from outside
            the main execution loop while preserving fee and PnL logging.

        Returns:
            Tuple containing:
                - pnl (float): Profit or loss realized from closing the position.
                - fees (float): Transaction fees incurred during closure.
        """
        return cls._close_position(
            fee_rate=fee_rate,
            position_state=position_state,
            price_x=price_x,
            price_y=price_y,
            exec_logger=exec_logger,
        )
