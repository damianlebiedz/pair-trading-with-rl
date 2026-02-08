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
        win: float,
        portfolio_value: float,
        total_fees: float,
        entry_threshold: float,
        exit_threshold: float,
        stop_loss_thr: float | None,
        delayed_entry: bool,
        time_stop: bool,
    ) -> tuple[float, float]:

        if prev_z_score is None:
            cls.long_signal = False
            cls.short_signal = False

        if delayed_entry:
            cls.long_signal = prev_z_score <= -entry_threshold
            cls.short_signal = prev_z_score >= entry_threshold

            cls.open_cond = (
                prev_z_score is not None
                and cls.short_signal
                and z_score < entry_threshold
            ) or (
                prev_z_score is not None
                and cls.long_signal
                and z_score > -entry_threshold
            )
        else:
            cls.long_signal = z_score <= -entry_threshold
            cls.short_signal = z_score >= entry_threshold

            cls.open_cond = (
                prev_z_score is not None
                and cls.short_signal
                and prev_z_score < entry_threshold
                and z_score <= stop_loss_thr
            ) or (
                prev_z_score is not None
                and cls.long_signal
                and prev_z_score > -entry_threshold
            )

        if cls.long_signal:
            cls.signal = 1
        elif cls.short_signal:
            cls.signal = -1
        else:
            cls.signal = 0

        # IN POSITION
        if position_state.prev_position != 0:
            # CLOSE POSITION (NO MEAN-REVERSION)
            if z_score is None:
                return cls._close_position(
                    ctx, position_state, price_x, price_y, total_fees
                )
            # CLOSE POSITION (TIME EXIT)
            elif time_stop and position_state.time_in_pos >= win:
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
                if (position_state.prev_position < 0 and cls.long_signal) or (
                    position_state.prev_position > 0 and cls.short_signal
                ):
                    pnl_close, total_fees_after_close = cls._close_position(
                        ctx, position_state, price_x, price_y, total_fees
                    )
                    _, total_fees_final = cls._open_position(
                        ctx,
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
            elif cls.open_cond:
                return cls._open_position(
                    ctx,
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
        beta: float,
        position_state: PositionState,
        price_x: float,
        price_y: float,
        total_fees: float,
        portfolio_value: float,
    ) -> tuple[float, float]:
        wx = 1 / (beta + 1)
        wy = beta / (beta + 1)

        x_spread, y_spread = cls.get_spread(
            ctx.ticker_x, ctx.ticker_y, position_state.position
        )

        if cls.long_signal:
            qx = portfolio_value * wx / (price_x * x_spread)
            qy = -(portfolio_value * wy) / (price_y * y_spread)
        elif cls.short_signal:
            qx = -(portfolio_value * wx) / (price_x * x_spread)
            qy = portfolio_value * wy / (price_y * y_spread)
        else:
            raise ValueError("Cannot open the position while 'position' is 0")

        entry_dif = qx * (price_x * x_spread) + qy * (price_y * y_spread)

        position_state.update_position(
            position=cls.signal,
            prev_position=position_state.prev_position,
            q_x=qx,
            q_y=qy,
            w_x=wx,
            w_y=wy,
            entry_dif=entry_dif,
        )

        pos_fees = portfolio_value * ctx.fee_rate
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
        position_state.time_in_pos += 1

        return pnl, total_fees
