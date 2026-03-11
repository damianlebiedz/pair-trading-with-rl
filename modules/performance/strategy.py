import numpy as np
import pandas as pd

from modules.core.enums import Interval, BetaHedge, Source
from modules.core.execution import TradeExecutor
from modules.core.indicators import (
    calculate_z_score,
    calculate_beta,
    generate_signal,
    calculate_spread_statistics,
    calculate_hurst,
)
from modules.data_services.data_loaders import load_pair
from modules.data_services.data_utils import add_log_prices
from modules.learning.agents import RLAgentAdapter
from modules.performance.models import (
    PositionState,
    StrategyResult,
    ExecLogger,
)
from modules.performance.stats import calculate_stats
from modules.learning.models import AgentState


class Strategy:
    """Main class for Strategy execution."""

    def __init__(
        self,
        ticker_x: str,
        ticker_y: str,
        start: str,
        end: str,
        interval: Interval,
        fee_rate: float,
        initial_cash: float,
        risk_free_rate_annual: float,
        beta_hedge: BetaHedge,
        delayed_entry: bool,
        sl_lock: bool,
        vol_window: int,
        time_decay_sl: bool,
        time_decay_params: tuple[int, int],
        freeze_std: bool,
        agent: RLAgentAdapter | None = None,
        source: Source = Source.LOG.value,
    ):
        self.ticker_x = ticker_x
        self.ticker_y = ticker_y
        self.start = start
        self.end = end
        self.interval = interval
        self.fee_rate = fee_rate
        self.initial_cash = initial_cash
        self.risk_free_rate_annual = risk_free_rate_annual
        self.beta_hedge = beta_hedge
        self.delayed_entry = delayed_entry
        self.sl_lock = sl_lock
        self.vol_window = vol_window
        self.time_decay_sl = time_decay_sl
        self.time_decay_params = time_decay_params
        self.freeze_std = freeze_std
        self.agent = agent
        self.source = source

        self.data = load_pair(
            x=self.ticker_x,
            y=self.ticker_y,
            start=self.start,
            end=self.end,
            interval=self.interval,
        )

        add_log_prices(self.data, self.ticker_x, self.ticker_y)

    def _execute_loop(
        self,
        df: pd.DataFrame,
        initial_cash: float,
        entry_threshold: float,
        exit_threshold: float,
        test_start: str,
        test_end: str,
        z_score_window: int,
        beta_test_start: str,
        stop_loss: float | None,
    ) -> pd.DataFrame:
        """
        Core backtesting loop that iterates through market data to simulate strategy execution.

        This method performs the following steps:
        1. Calculates static parameters (Beta, Window) based on pre-test data.
        2. Computes market volatility features for risk assessment.
        3. Iterates bar-by-bar through the `test_start` to `test_end` range.
        4. Calculates dynamic indicators (Z-Score, Spread) inside the loop.
        5. Generates signals and executes trades via TradeExecutor.
        6. Tracks equity, PnL, fees, and drawdown state.
        7. Handles stop-loss logic (including time-based decay if enabled).
        8. Force-closes any open positions at the end of the simulation period.

        Note:
            - Z-Score window is calculated with the current close included. More in research paper.

        Returns:
            tuple[pd.DataFrame, pd.DataFrame]:
                1. Strategy DataFrame: Time-series data containing price, equity curve,
                   signals, z-scores, PnL per step, and drawdown.
                2. Execution Logger DataFrame: Detailed record of individual trades (entries/exits),
                   fees and execution prices.
        """
        df = df.copy()

        x_col = self.ticker_x
        y_col = self.ticker_y
        source_x_col = f"{x_col}_{self.source}"
        source_y_col = f"{y_col}_{self.source}"

        offset = pd.Timedelta(self.interval.value)

        beta_start_dt = pd.to_datetime(beta_test_start)
        test_start_dt = pd.to_datetime(test_start) + offset
        test_end_dt = pd.to_datetime(test_end)

        beta_start_pos = df.index.get_indexer([beta_start_dt], method="bfill")[0]
        test_start_pos = df.index.get_indexer([test_start_dt], method="bfill")[0]
        end_pos = df.index.get_indexer([test_end_dt], method="ffill")[0]

        if -1 in [beta_start_pos, test_start_pos, end_pos]:
            raise KeyError(
                f"Index out of bounds! \n"
                f"Requested -> Beta: {beta_start_dt}, Test: {test_start_dt}, End: {test_end_dt}\n"
                f"Available -> First: {df.index[0]}, Last: {df.index[-1]}"
            )

        X_vals = df[source_x_col].values
        Y_vals = df[source_y_col].values
        N = len(df)

        beta_lookback_len = test_start_pos - beta_start_pos + 1

        if beta_lookback_len < 1:
            raise ValueError("'beta_test_start' cannot be later than 'test_start'")
        if test_start_pos - z_score_window + 1 < 1:
            raise ValueError("'z_score_window' cannot be later than 'test_start'")

        if self.beta_hedge == "no_hedge":
            market_beta = 1.0
        else:
            slice_x_warmup_beta = X_vals[beta_start_pos : test_start_pos + 1]
            slice_y_warmup_beta = Y_vals[beta_start_pos : test_start_pos + 1]
            market_beta = calculate_beta(
                X_slice=slice_x_warmup_beta,
                Y_slice=slice_y_warmup_beta,
            )

        precalc_ols_beta = np.zeros(N)
        if self.beta_hedge == "rolling":
            cov = pd.Series(X_vals).rolling(beta_lookback_len).cov(pd.Series(Y_vals))
            var = pd.Series(Y_vals).rolling(beta_lookback_len).var()
            precalc_ols_beta = np.nan_to_num((cov / var).values, nan=0.0)

        precalc_hurst_free = np.full(N, 0.5)

        if self.beta_hedge == "no_hedge":
            base_beta_arr = np.ones(N)
        elif self.beta_hedge == "static":
            base_beta_arr = np.full(N, market_beta)
        else:
            base_beta_arr = precalc_ols_beta

        for i in range(test_start_pos, N):
            b = base_beta_arr[i]

            slice_x_beta = X_vals[i - beta_lookback_len + 1 : i + 1]
            slice_y_beta = Y_vals[i - beta_lookback_len + 1 : i + 1]

            precalc_hurst_free[i] = calculate_hurst(
                X_slice=slice_x_beta, Y_slice=slice_y_beta, beta=b
            )

        df[f"ret_{self.ticker_x}"] = df[source_x_col].diff().fillna(0.0)
        df[f"ret_{self.ticker_y}"] = df[source_y_col].diff().fillna(0.0)

        vol_window = self.vol_window
        df[f"vol_{self.ticker_x}"] = (
            df[f"ret_{self.ticker_x}"].rolling(window=vol_window).std()
        )
        df[f"vol_{self.ticker_y}"] = (
            df[f"ret_{self.ticker_y}"].rolling(window=vol_window).std()
        )

        df["market_vol"] = (
            df[f"vol_{self.ticker_x}"] + df[f"vol_{self.ticker_y}"]
        ) / 2.0
        df["market_vol"] = df["market_vol"].fillna(0.0)

        equity = initial_cash
        equity_peak = initial_cash

        total_pnl = 0.0
        total_fees = 0.0
        position_state = PositionState()
        exec_logger = ExecLogger()

        prev_z_score = None

        if stop_loss is not None:
            stop_loss_thr = entry_threshold * stop_loss
        else:
            stop_loss_thr = None

        is_bankrupt = False
        results_buffer = []

        for i in range(test_start_pos, len(df)):
            price_x = df[x_col].iloc[i]
            price_y = df[y_col].iloc[i]
            idx = df.index[i]

            z_score = spread = mean = std = None
            market_z_score = market_spread = market_mean = market_std = None
            market_hurst = None

            if is_bankrupt:
                total_net_pnl = -initial_cash
                equity = 0.0
                drawdown_pct = -1.0
                signal = 0
            else:
                if self.beta_hedge == "rolling":
                    market_beta = base_beta_arr[i]

                if self.time_decay_sl and stop_loss_thr is not None:
                    time_decay_start = self.time_decay_params[0]
                    time_decay_end = self.time_decay_params[1]

                    hl_diff = (time_decay_end * z_score_window) - (
                        time_decay_start * z_score_window
                    )
                    sl_exit_diff = stop_loss_thr - exit_threshold
                    decay_per_iter = (
                        sl_exit_diff / hl_diff if hl_diff != 0.0 else sl_exit_diff
                    )
                else:
                    time_decay_start = 0.0
                    decay_per_iter = 0.0

                if (
                    position_state.position != 0
                    and position_state.entry_beta is not None
                ):
                    beta = position_state.entry_beta
                else:
                    beta = market_beta

                if beta <= 0:
                    pass
                else:
                    slice_x_win = X_vals[i - z_score_window + 1 : i + 1]
                    slice_y_win = Y_vals[i - z_score_window + 1 : i + 1]

                    market_spread, market_mean, market_std = (
                        calculate_spread_statistics(
                            X_slice=slice_x_win,
                            Y_slice=slice_y_win,
                            beta=market_beta,
                        )
                    )
                    market_z_score = calculate_z_score(
                        spread=market_spread, mean=market_mean, std=market_std
                    )

                    if (
                        position_state.position != 0
                        and position_state.entry_beta is not None
                    ):
                        spread, mean, current_std = calculate_spread_statistics(
                            X_slice=slice_x_win,
                            Y_slice=slice_y_win,
                            beta=position_state.entry_beta,
                        )

                        std = (
                            position_state.entry_std if self.freeze_std else current_std
                        )

                        z_score = calculate_z_score(spread=spread, mean=mean, std=std)

                    else:
                        z_score = market_z_score
                        spread = market_spread
                        mean = market_mean
                        std = market_std

                signal = generate_signal(
                    z_score=z_score,
                    prev_z_score=prev_z_score,
                    entry_threshold=entry_threshold,
                    stop_loss_thr=stop_loss_thr,
                    delayed_entry=self.delayed_entry,
                )

                if position_state.position == 0:
                    position_state.sl_thr = stop_loss_thr
                elif (
                    self.time_decay_sl
                    and stop_loss_thr is not None
                    and position_state.time_in_pos >= time_decay_start * z_score_window
                ):
                    position_state.sl_thr -= decay_per_iter

                if equity > equity_peak:
                    equity_peak = equity

                if equity_peak > 0:
                    drawdown_pct = (equity - equity_peak) / equity_peak
                else:
                    drawdown_pct = 0.0

                if position_state.sl_lock:
                    if (
                        z_score is not None
                        and prev_z_score is not None
                        and exit_threshold is not None
                    ):
                        break_above = prev_z_score > exit_threshold >= z_score
                        break_below = prev_z_score < -exit_threshold <= z_score
                        if break_above or break_below:
                            position_state.sl_lock = False

                market_hurst = precalc_hurst_free[i]

                action, sl_lock = TradeExecutor.decide(
                    position_state=position_state,
                    signal=signal,
                    z_score=z_score,
                    exit_threshold=exit_threshold,
                )
                if sl_lock and self.sl_lock:
                    position_state.sl_lock = True

                if self.agent:
                    if z_score is None or pd.isna(z_score):
                        action = 0.0
                    else:
                        current_state = AgentState(
                            z_score=z_score,
                            market_beta=market_beta,
                            hurst=market_hurst,
                            position=position_state.position,
                            signal=action,
                            norm_time_in_pos=position_state.time_in_pos
                            / z_score_window,
                            drawdown_pct=drawdown_pct,
                            current_market_vol=df["market_vol"].iloc[i],
                        )
                        action = self.agent.get_action(current_state)

                position_state.open_time = idx

                pnl, fees = TradeExecutor.execute(
                    fee_rate=self.fee_rate,
                    position_state=position_state,
                    action=action,
                    stop_loss_thr=stop_loss_thr,
                    price_x=price_x,
                    price_y=price_y,
                    beta=beta,
                    equity=equity,
                    exec_logger=exec_logger,
                    std=std,
                    sl_lock=position_state.sl_lock,
                )

                prev_z_score = z_score

                total_pnl += pnl
                total_fees += fees
                total_net_pnl = total_pnl - total_fees

                equity = initial_cash + total_net_pnl

                if equity < 0.0:
                    is_bankrupt = True
                    position_state.clear_position()
                    equity = 0.0
                    drawdown_pct = -1.0
                    total_net_pnl = -initial_cash
                    if total_pnl < -initial_cash:
                        total_pnl = -initial_cash

            results_buffer.append(
                {
                    "index": idx,
                    "z_score": z_score,
                    "spread": spread,
                    "mean": mean,
                    "std": std,
                    "beta": position_state.entry_beta or market_beta,
                    "market_z_score": market_z_score,
                    "market_spread": market_spread,
                    "market_mean": market_mean,
                    "market_std": market_std,
                    "market_beta": market_beta,
                    "hurst": market_hurst,
                    "window": z_score_window,
                    "entry_thr": entry_threshold,
                    "exit_thr": exit_threshold,
                    "sl_thr": position_state.sl_thr,
                    "sl_lock": int(position_state.sl_lock),
                    "q_x": position_state.q_x,
                    "q_y": position_state.q_y,
                    "w_x": position_state.w_x,
                    "w_y": position_state.w_y,
                    "signal": signal,
                    "position": position_state.position,
                    "equity": equity,
                    "total_pnl": total_pnl,
                    "total_fees": total_fees,
                    "total_net_pnl": total_net_pnl,
                    "total_return": total_pnl / initial_cash,
                    "total_net_return": total_net_pnl / initial_cash,
                    "drawdown_pct": drawdown_pct,
                }
            )

            position_state.prev_position = position_state.position

        if results_buffer:
            if position_state.position != 0:
                price_x = df[x_col].iloc[-1]
                price_y = df[y_col].iloc[-1]

                pnl, fees = TradeExecutor.call_close_position(
                    fee_rate=self.fee_rate,
                    position_state=position_state,
                    price_x=price_x,
                    price_y=price_y,
                    exec_logger=exec_logger,
                )

                results_buffer[-1]["total_fees"] += fees
                results_buffer[-1]["total_net_pnl"] += pnl - fees
                results_buffer[-1]["q_x"] = 0
                results_buffer[-1]["q_y"] = 0
                results_buffer[-1]["w_x"] = None
                results_buffer[-1]["w_y"] = None
                results_buffer[-1]["position"] = 0

            results_df = pd.DataFrame(results_buffer)
            results_df.set_index("index", inplace=True)
            df.loc[results_df.index, results_df.columns] = results_df

            last_processed_idx = results_buffer[-1]["index"]
            last_pos_loc = df.index.get_loc(last_processed_idx)
            final_slice_end = min(last_pos_loc, end_pos)

            warmup_start = test_start_pos - z_score_window + 1
            df = df.iloc[warmup_start : final_slice_end + 1].copy()
        else:
            warmup_start = test_start_pos - z_score_window + 1
            df = df.iloc[warmup_start : end_pos + 1].copy()

        exec_log_df = exec_logger.to_df()
        exec_log_df["ticker"] = self.ticker_x + "-" + self.ticker_y

        return (
            df.drop(
                columns=[
                    f"ret_{self.ticker_x}",
                    f"ret_{self.ticker_y}",
                    f"vol_{self.ticker_x}",
                    f"vol_{self.ticker_y}",
                ],
                errors="ignore",
            ),
            exec_log_df,
        )

    def run_strategy(
        self,
        z_score_window: int,
        entry_threshold: float,
        exit_threshold: float | None,
        stop_loss: float | None,
        test_start: str,
        test_end: str,
        beta_test_start: str,
    ) -> StrategyResult:
        """
        Executes the strategy backtest with specific parameters.

        Returns:
            StrategyResult: Object containing backtest data, performance statistics and execution logger.
        """

        data, exec_log_df = self._execute_loop(
            df=self.data,
            initial_cash=self.initial_cash,
            entry_threshold=entry_threshold,
            exit_threshold=exit_threshold,
            test_start=test_start,
            test_end=test_end,
            z_score_window=z_score_window,
            beta_test_start=beta_test_start,
            stop_loss=stop_loss,
        )

        stats = calculate_stats(
            df=data,
            exec_log_df=exec_log_df,
            initial_cash=self.initial_cash,
            interval=self.interval,
            risk_free_rate_annual=self.risk_free_rate_annual,
        )

        return StrategyResult(
            data=data,
            ticker_x=self.ticker_x,
            ticker_y=self.ticker_y,
            start=test_start,
            end=test_end,
            interval=self.interval,
            fee_rate=self.fee_rate,
            stats=stats,
            exec_logger=exec_log_df,
        )
