from modules.core.models import ExecutionContext, PositionState, PositionContext


class TradeExecutor:
    @classmethod
    def execute(
        cls,
        ctx: ExecutionContext,
        pos_ctx: PositionContext,
        position_state: PositionState,
        action: float,
        price_x: float,
        price_y: float,
        z_score: float | None,
        beta: float,
        portfolio_value: float,
        exit_threshold: float,
    ) -> tuple[float, float]:
        prev_position = position_state.prev_position
        stop_loss_thr = position_state.sl_thr

        # IN POSITION
        if prev_position != 0:
            # CLOSE POSITION (NO MEAN-REVERSION OR ACTION = 0)
            if z_score is None or action == 0:
                return cls._close_position(
                    ctx, position_state, price_x, price_y
                )
            # CLOSE POSITION (STOP LOSS OR TAKE PROFIT FROM SHORT LEG)
            elif (
                prev_position < 0
                and (
                    z_score <= exit_threshold
                    or (stop_loss_thr is not None and z_score >= stop_loss_thr)
                )
                # CLOSE POSITION (STOP LOSS OR TAKE PROFIT FROM LONG LEG)
            ) or (
                prev_position > 0
                and (
                    z_score >= -exit_threshold
                    or (stop_loss_thr is not None and z_score <= -stop_loss_thr)
                )
            ):
                # OPEN REVERSE POSITION
                if (prev_position < 0 < action) or (
                    prev_position > 0 > action
                ):
                    pnl_close, fees_after_close = cls._close_position(
                        ctx, position_state, price_x, price_y
                    )
                    _, fees_after_open = cls._open_position(
                        ctx,
                        pos_ctx,
                        action,
                        beta,
                        position_state,
                        price_x,
                        price_y,
                        portfolio_value,
                    )
                    return pnl_close, fees_after_close + fees_after_open

                return cls._close_position(
                    ctx, position_state, price_x, price_y
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
                    ctx,
                    pos_ctx,
                    action,
                    beta,
                    position_state,
                    price_x,
                    price_y,
                    portfolio_value,
                )
            # STAY OUT OF POSITION
            else:
                return 0.0, 0.0

    @classmethod
    def _open_position(
        cls,
        ctx: ExecutionContext,
        pos_ctx: PositionContext,
        action: float,
        beta: float,
        position_state: PositionState,
        price_x: float,
        price_y: float,
        portfolio_value: float,
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
            prev_dif=entry_dif,
            sl_thr=pos_ctx.base_sl_thr,
        )

        fees = pos_cash * ctx.fee_rate

        return 0.0, fees

    @classmethod
    def _close_position(
        cls, ctx, position_state, price_x, price_y
    ) -> tuple[float, float]:
        exit_dif = position_state.q_x * price_x + position_state.q_y * price_y
        exit_val = abs(position_state.q_x) * price_x + abs(position_state.q_y * price_y)

        pnl = exit_dif - position_state.prev_dif
        fees = exit_val * ctx.fee_rate

        position_state.clear_position()

        return pnl, fees

    @staticmethod
    def _hold_position(
        position_state, price_x, price_y
    ) -> tuple[float, float]:
        curr_dif = position_state.q_x * price_x + position_state.q_y * price_y

        position_state.position = position_state.prev_position
        position_state.time_in_pos += 1

        pnl = curr_dif - position_state.prev_dif
        position_state.prev_dif = curr_dif

        return pnl, 0.0
