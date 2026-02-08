from modules.core.models import ExecutionContext, PositionState


class TradeExecutor:
    @classmethod
    def execute(
        cls,
        ctx: ExecutionContext,
        position_state: PositionState,
        action: float,
        price_x: float,
        price_y: float,
        z_score: float | None,
        beta: float,
        portfolio_value: float,
        total_fees: float,
        exit_threshold: float,
    ) -> tuple[float, float]:
        stop_loss_thr = position_state.sl_thr

        # IN POSITION
        if position_state.prev_position != 0:
            # CLOSE POSITION (NO MEAN-REVERSION OR ACTION = 0)
            if z_score is None or action == 0:
                return cls._close_position(
                    ctx, position_state, price_x, price_y, total_fees
                )
            # CLOSE POSITION (STOP LOSS OR TAKE PROFIT FROM SHORT LEG)
            elif (
                position_state.prev_position < 0
                and (
                    z_score <= exit_threshold
                    or (stop_loss_thr is not None and z_score >= stop_loss_thr)
                )
                # CLOSE POSITION (STOP LOSS OR TAKE PROFIT FROM LONG LEG)
            ) or (
                position_state.prev_position > 0
                and (
                    z_score >= -exit_threshold
                    or (stop_loss_thr is not None and z_score <= -stop_loss_thr)
                )
            ):
                # OPEN REVERSE POSITION
                if (position_state.prev_position < 0 < action) or (
                    position_state.prev_position > 0 > action
                ):
                    pnl_close, total_fees_after_close = cls._close_position(
                        ctx, position_state, price_x, price_y, total_fees
                    )
                    _, total_fees_final = cls._open_position(
                        ctx,
                        action,
                        beta,
                        position_state,
                        price_x,
                        price_y,
                        total_fees_after_close,
                        portfolio_value,
                    )
                    return pnl_close, total_fees_final

                return cls._close_position(
                    ctx, position_state, price_x, price_y, total_fees
                )
            # HOLD POSITION
            else:
                return cls._hold_position(position_state, price_x, price_y, total_fees)

        # OUT OF POSITION
        else:
            # STAY OUT OF POSITION
            if z_score is None:
                return 0, total_fees
            # OPEN POSITION
            elif action != 0:
                return cls._open_position(
                    ctx,
                    action,
                    beta,
                    position_state,
                    price_x,
                    price_y,
                    total_fees,
                    portfolio_value,
                )
            # STAY OUT OF POSITION
            else:
                return 0, total_fees

    @classmethod
    def _open_position(
        cls,
        ctx: ExecutionContext,
        action: float,
        beta: float,
        position_state: PositionState,
        price_x: float,
        price_y: float,
        total_fees: float,
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
            prev_position=position_state.prev_position,
            q_x=qx,
            q_y=qy,
            w_x=wx,
            w_y=wy,
            entry_dif=entry_dif,
        )

        pos_fees = pos_cash * ctx.fee_rate
        t_fees = total_fees + pos_fees

        return 0, t_fees

    @classmethod
    def _close_position(
        cls, ctx, position_state, price_x, price_y, total_fees
    ) -> tuple[float, float]:
        exit_dif = position_state.q_x * price_x + position_state.q_y * price_y
        exit_val = abs(position_state.q_x) * price_x + abs(position_state.q_y * price_y)
        pos_fees = exit_val * ctx.fee_rate

        if position_state.prev_position != 0:
            pnl = exit_dif - position_state.entry_dif
        else:
            raise ValueError("Cannot close the position while 'position' is 0")

        position_state.clear_position()
        t_fees = total_fees + pos_fees

        return pnl, t_fees

    @staticmethod
    def _hold_position(
        position_state, price_x, price_y, total_fees
    ) -> tuple[float, float]:
        curr_dif = position_state.q_x * price_x + position_state.q_y * price_y

        if position_state.prev_position != 0:
            pnl = curr_dif - position_state.entry_dif
        else:
            raise ValueError("Cannot hold the position while 'position' is 0")

        position_state.position = position_state.prev_position
        position_state.time_in_pos += 1

        return pnl, total_fees
