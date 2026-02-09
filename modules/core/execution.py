from modules.core.models import (
    ExecutionContext,
    PositionState,
    StrategyContext,
    Log,
    ExecLogger,
)


class TradeExecutor:
    @classmethod
    def execute(
        cls,
        exec_ctx: ExecutionContext,
        str_ctx: StrategyContext,
        position_state: PositionState,
        action: float,
        price_x: float,
        price_y: float,
        z_score: float | None,
        beta: float,
        portfolio_value: float,
        exec_logger: ExecLogger,
    ) -> tuple[float, float]:
        prev_position = position_state.prev_position
        stop_loss_thr = position_state.sl_thr

        # IN POSITION
        if prev_position != 0:
            # CLOSE POSITION (NO MEAN-REVERSION OR ACTION = 0)
            if z_score is None or action == 0:
                return cls._close_position(
                    exec_ctx, position_state, price_x, price_y, exec_logger
                )
            # CLOSE POSITION (STOP LOSS OR TAKE PROFIT FROM SHORT LEG)
            elif (
                prev_position < 0
                and (
                    z_score <= str_ctx.exit_threshold
                    or (stop_loss_thr is not None and z_score >= stop_loss_thr)
                )
                # CLOSE POSITION (STOP LOSS OR TAKE PROFIT FROM LONG LEG)
            ) or (
                prev_position > 0
                and (
                    z_score >= -str_ctx.exit_threshold
                    or (stop_loss_thr is not None and z_score <= -stop_loss_thr)
                )
            ):
                # OPEN REVERSE POSITION
                if (prev_position < 0 < action) or (prev_position > 0 > action):
                    pnl_close, fees_after_close = cls._close_position(
                        exec_ctx, position_state, price_x, price_y, exec_logger
                    )
                    _, fees_after_open = cls._open_position(
                        exec_ctx,
                        str_ctx,
                        action,
                        beta,
                        position_state,
                        price_x,
                        price_y,
                        portfolio_value,
                        exec_logger,
                    )
                    return pnl_close, fees_after_close + fees_after_open

                return cls._close_position(
                    exec_ctx, position_state, price_x, price_y, exec_logger
                )
            # HOLD POSITION
            else:
                return cls._hold_position(position_state, price_x, price_y)

        # OUT OF POSITION
        else:
            # STAY OUT OF POSITION
            if z_score is None:
                return 0.0, 0.0
            # OPEN POSITION
            elif action != 0:
                return cls._open_position(
                    exec_ctx,
                    str_ctx,
                    action,
                    beta,
                    position_state,
                    price_x,
                    price_y,
                    portfolio_value,
                    exec_logger,
                )
            # STAY OUT OF POSITION
            else:
                return 0.0, 0.0

    @classmethod
    def _open_position(
        cls,
        ctx: ExecutionContext,
        str_ctx: StrategyContext,
        action: float,
        beta: float,
        position_state: PositionState,
        price_x: float,
        price_y: float,
        portfolio_value: float,
        exec_logger: ExecLogger,
    ) -> tuple[float, float]:
        wx = 1 / (beta + 1)
        wy = beta / (beta + 1)

        pos_cash = portfolio_value * abs(action)

        if action > 0:
            qx = pos_cash * wx / price_x
            qy = -(pos_cash * wy) / price_y
        elif action < 0:
            qx = -(pos_cash * wx) / price_x
            qy = pos_cash * wy / price_y
        else:
            raise ValueError("Cannot open the position while signal == 0")

        entry_dif = qx * price_x + qy * price_y

        position_state.update_position(
            position=action,
            q_x=qx,
            q_y=qy,
            w_x=wx,
            w_y=wy,
            entry_dif=entry_dif,
            prev_dif=entry_dif,
            sl_thr=str_ctx.sl_threshold,
        )

        fees = pos_cash * ctx.fee_rate

        log = Log(
            open_time=position_state.open_time,
            price_x=price_x,
            price_y=price_y,
            qx=qx,
            qy=qy,
            position=position_state.position,
            fees=fees,
        )
        exec_logger.append(log)

        return 0.0, fees

    @classmethod
    def _close_position(
        cls,
        ctx: ExecutionContext,
        position_state: PositionState,
        price_x: float,
        price_y: float,
        exec_logger: ExecLogger,
    ) -> tuple[float, float]:
        exit_dif = position_state.q_x * price_x + position_state.q_y * price_y
        exit_val = abs(position_state.q_x) * price_x + abs(position_state.q_y * price_y)

        pnl = exit_dif - position_state.prev_dif
        fees = exit_val * ctx.fee_rate

        log = Log(
            open_time=position_state.open_time,
            price_x=price_x,
            price_y=price_y,
            qx=0.0,
            qy=0.0,
            position=0.0,
            fees=fees,
            pnl=exit_dif - position_state.entry_dif,
            time_in_pos=position_state.time_in_pos,
        )
        exec_logger.append(log)

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
        ctx: ExecutionContext,
        position_state: PositionState,
        price_x: float,
        price_y: float,
        exec_logger: ExecLogger,
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
        return cls._close_position(ctx, position_state, price_x, price_y, exec_logger)
