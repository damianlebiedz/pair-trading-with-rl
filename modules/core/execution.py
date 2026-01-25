from modules.core.models import ExecutionContext, PositionState


class TradeExecutor:
    @staticmethod
    def get_spread(x: str, y: str, position: float) -> tuple[float, float]:  # TODO
        """Get spread for two assets depending on position."""
        if position == 0:
            # SPREAD FOR POSITION CLOSING
            return 1.0, 1.0
        elif position > 0:
            # SPREAD FOR POSITIVE POSITION OPENING
            return 1.0, 1.0
        else:
            # SPREAD FOR NEGATIVE POSITION OPENING
            return 1.0, 1.0

    @classmethod
    def execute(
        cls,
        ctx: ExecutionContext,
        position_state: PositionState,
        price_x: float,
        price_y: float,
        z_score: float | None,
        prev_z_score: float | None,
        beta: float,
        total_fees: float,
        entry_threshold: float,
        exit_threshold: float,
        stop_loss: float,
    ) -> tuple[float, float]:

        # IN POSITION
        if position_state.prev_position != 0:
            # CLOSE POSITION (STOP LOSS OR TAKE PROFIT)
            if z_score is None:
                # NO MEAN-REVERSION (FROM HALF-LIFE WINDOW CALCULATION) OR BETA < 0
                return cls._close_position(
                    ctx, position_state, price_x, price_y, total_fees
                )
            elif (
                position_state.prev_position < 0
                and (
                    z_score <= exit_threshold
                    or (
                        position_state.stop_loss_threshold is not None
                        and z_score >= position_state.stop_loss_threshold
                    )
                )
            ) or (
                position_state.prev_position > 0
                and (
                    z_score >= -exit_threshold
                    or (
                        position_state.stop_loss_threshold is not None
                        and z_score <= -position_state.stop_loss_threshold
                    )
                )
            ):
                # OPEN REVERSE POSITION
                if (position_state.prev_position < 0 < position_state.signal) or (
                    position_state.prev_position > 0 > position_state.signal
                ):
                    pnl_close, total_fees_after_close = cls._close_position(
                        ctx, position_state, price_x, price_y, total_fees
                    )
                    _, total_fees_final = cls._open_position(
                        ctx,
                        beta,
                        z_score,
                        position_state,
                        price_x,
                        price_y,
                        total_fees_after_close,
                        stop_loss,
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
            elif (prev_z_score < entry_threshold and position_state.signal < 0) or (
                prev_z_score > -entry_threshold and position_state.signal > 0
            ):
                return cls._open_position(
                    ctx,
                    beta,
                    z_score,
                    position_state,
                    price_x,
                    price_y,
                    total_fees,
                    stop_loss,
                )
            # STAY OUT OF POSITION
            else:
                return 0, total_fees

    @classmethod
    def _open_position(
        cls,
        ctx,
        beta,
        z_score,
        position_state,
        price_x,
        price_y,
        total_fees,
        stop_loss,
    ) -> tuple[float, float]:
        wx = 1 / (beta + 1)
        wy = beta / (beta + 1)

        x_spread, y_spread = cls.get_spread(
            ctx.ticker_x, ctx.ticker_y, position_state.position
        )

        if position_state.signal > 0:
            qx = ctx.initial_cash * wx / (price_x * x_spread)   # TODO: tutaj initial_cash * position jeśli position != |1|
            qy = -(ctx.initial_cash * wy) / (price_y * y_spread)
        elif position_state.signal < 0:
            qx = -(ctx.initial_cash * wx) / (price_x * x_spread)
            qy = ctx.initial_cash * wy / (price_y * y_spread)
        else:
            raise ValueError("Cannot open the position while 'position' is 0")

        if stop_loss is not None:
            stop_loss_thr = abs(z_score * stop_loss)
        else:
            stop_loss_thr = None

        entry_dif = qx * (price_x * x_spread) + qy * (price_y * y_spread)

        position_state.update_position(
            position=position_state.signal,
            prev_position=position_state.prev_position,
            q_x=qx,
            q_y=qy,
            w_x=wx,
            w_y=wy,
            stop_loss_threshold=stop_loss_thr,
            entry_dif=entry_dif,
        )

        pos_fees = ctx.initial_cash * ctx.fee_rate
        t_fees = total_fees + pos_fees

        return 0, t_fees

    @classmethod
    def _close_position(
        cls, ctx, position_state, price_x, price_y, total_fees
    ) -> tuple[float, float]:
        x_spread, y_spread = cls.get_spread(ctx.ticker_x, ctx.ticker_y, 0)

        exit_dif = position_state.q_x * (price_x * x_spread) + position_state.q_y * (
            price_y * y_spread
        )
        exit_val = abs(position_state.q_x) * (price_x * x_spread) + abs(
            position_state.q_y
        ) * (price_y * y_spread)
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

        return pnl, total_fees
