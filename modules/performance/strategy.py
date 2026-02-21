from typing import Literal
import numpy as np
import pandas as pd

from modules.core.execution import TradeExecutor
from modules.core.indicators import (
    calculate_z_score,
    calculate_beta,
    calculate_half_life_window,
    generate_signal,
    KalmanState,
    calculate_spread_statistics,
    calculate_hurst,
)
from modules.data_services.data_loaders import load_pair
from modules.data_services.data_utils import add_log_prices
from modules.performance.models import (
    PositionState,
    StrategyResult,
    ExecLogger,
)
from modules.learning.agents import RLAgentAdapter
from modules.performance.stats import calculate_stats
from modules.learning.models import AgentState


class Strategy:
    """
    Main class for Strategy execution.

    Args:
        ticker_x (str): Ticker for asset X.
        ticker_y (str): Ticker for asset Y.
        start (str): Data start date.
        end (str): Data end date.
        interval (str): Data timeframe.
        fee_rate (float): Transaction fee rate (e.g., 0.001 for 0.1%).
        initial_cash (float): Starting capital (the same for every trade).
        risk_free_rate_annual (float): Annual risk-free rate.
        beta_hedge (str): Hedge ratio mode: "static", "rolling", or "no_hedge".
        beta_method (str): Beta calculation method: "ols" or "kalman".
        delayed_entry (bool): Delayed execution or standard one.
        time_decay_sl (bool): Time Decay SL: if true, SL decay will start from 0.5 * window and be equal to exit threshold at window size.
        agent (RLAgentAdapter): RL Agent, if None - trade without agent, otherwise - use agent's actions.
        vol_window (int): Volatility window size. Default = 24 (one day in '1h' interval).
        valid_window (tuple(int, int)): Min and max Z-Score window.
        source (str): Type of prices. Default = "log".
    """

    def __init__(
        self,
        ticker_x: str,
        ticker_y: str,
        start: str,
        end: str,
        interval: Literal["1d", "4h", "1h", "30m", "15m", "5m", "3m", "1m"],
        fee_rate: float,
        initial_cash: float,
        risk_free_rate_annual: float,
        beta_hedge: Literal["static", "rolling", "no_hedge"],
        beta_method: Literal["ols", "kalman"],
        window_method: Literal["fixed", "static", "rolling"],
        delayed_entry: bool,
        sl_lock: bool,
        vol_window: int,
        valid_window: tuple[int, int],
        time_decay_sl: bool,
        agent: RLAgentAdapter | None = None,
        source: str = "log",
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
        self.beta_method = beta_method
        self.window_method = window_method
        self.delayed_entry = delayed_entry
        self.sl_lock = sl_lock
        self.time_decay_sl = time_decay_sl
        self.agent = agent
        self.vol_window = vol_window
        self.valid_window = valid_window
        self.source = source

        if beta_hedge not in ["static", "rolling", "no_hedge"]:
            raise ValueError(
                "Invalid beta_hedge: should be 'static', 'rolling', or 'no_hedge'"
            )

        if beta_method not in ["ols", "kalman"]:
            raise ValueError("Invalid beta_method: should be 'ols' or 'kalman'")

        if window_method not in ["fixed", "static", "rolling"]:
            raise ValueError(
                "Invalid window_method: should be 'fixed' or 'static' or 'rolling'"
            )

        if valid_window[0] > valid_window[1]:
            raise ValueError(f"'valid_window' should be (min, max): {valid_window}")

        self.data = load_pair(
            x=ticker_x, y=ticker_y, start=start, end=end, interval=interval
        )

        add_log_prices(self.data, self.ticker_x, self.ticker_y)

    def _execute_loop(
        self,
        df: pd.DataFrame,
        initial_cash: float,
        entry_threshold: float | None,
        exit_threshold: float | None,
        test_start: str,
        test_end: str,
        fixed_window: int | None,
        win_test_start: str,
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

        Args:
            df (pd.DataFrame): DataFrame containing price data (columns must match ticker names).
            initial_cash (float): Starting capital for the simulation.
            entry_threshold (float | None): Z-score threshold for entering positions (long/short spread).
            exit_threshold (float | None): Z-score threshold for exiting positions (reversion to mean).
            test_start (str): Start date string (YYYY-MM-DD) for the backtest loop.
            test_end (str): End date string (YYYY-MM-DD) for the backtest loop.
            fixed_window (int | None): Parameter determining the lookback window size for 'fixed' window method.
            win_test_start (str): Start date for data used to calculate the initial window/beta.
            stop_loss (float | None): Stop-loss distance from entry threshold. None to disable.

        Returns:
            tuple[pd.DataFrame, pd.DataFrame]:
                1. Strategy DataFrame: Time-series data containing price, equity curve,
                   signals, z-scores, PnL per step, and drawdown.
                2. Execution Logger DataFrame: Detailed record of individual trades (entries/exits),
                   fees and execution prices.
        """
        if self.window_method == "fixed" and fixed_window is None:
            raise ValueError(
                "'fixed_window' should be integer when 'window_method' is 'fixed'"
            )

        df = df.copy()

        x_col = self.ticker_x
        y_col = self.ticker_y
        source_x_col = f"{x_col}_{self.source}"
        source_y_col = f"{y_col}_{self.source}"

        beta_method = self.beta_method

        test_start_pos = df.index.get_indexer(
            [pd.to_datetime(test_start)], method="bfill"
        )[0]
        target_date = pd.to_datetime(win_test_start)
        win_start_pos = df.index.get_indexer([target_date], method="bfill")[0]
        if df.index[win_start_pos] == target_date:
            win_start_pos += 1
        end_pos = df.index.get_indexer([pd.to_datetime(test_end)], method="bfill")[0]

        if -1 in [test_start_pos, win_start_pos, end_pos]:
            raise KeyError("Index not found in dataframe")

        X_vals = df[source_x_col].values
        Y_vals = df[source_y_col].values
        N = len(df)
        lookback_len = test_start_pos - win_start_pos + 1

        if self.beta_hedge == "no_hedge":
            market_beta = 1.0
        else:
            slice_x_warmup = X_vals[win_start_pos : test_start_pos + 1]
            slice_y_warmup = Y_vals[win_start_pos : test_start_pos + 1]

            market_beta = calculate_beta(
                X_slice=slice_x_warmup,
                Y_slice=slice_y_warmup,
                beta_method=beta_method,
            )

        kf_state = None
        if self.beta_method == "kalman" and self.beta_hedge == "rolling":
            kf_state = KalmanState()
            warmup_data = df.iloc[win_start_pos : test_start_pos + 1]
            for i in range(len(warmup_data)):
                obs_x = warmup_data[source_y_col].iloc[i]
                obs_y = warmup_data[source_x_col].iloc[i]
                kf_state.update(obs_x, obs_y)

        if self.window_method == "fixed":
            market_win = fixed_window
        else:
            slice_x_warmup = X_vals[win_start_pos : test_start_pos + 1]
            slice_y_warmup = Y_vals[win_start_pos : test_start_pos + 1]
            market_win = calculate_half_life_window(
                slice_x_warmup, slice_y_warmup, market_beta, self.valid_window
            )

        precalc_ols_beta = np.zeros(N)
        if self.beta_method == "ols" and self.beta_hedge == "rolling":
            cov = pd.Series(X_vals).rolling(lookback_len).cov(pd.Series(Y_vals))
            var = pd.Series(Y_vals).rolling(lookback_len).var()
            precalc_ols_beta = np.nan_to_num((cov / var).values, nan=0.0)

        precalc_kalman_beta = np.zeros(N)
        if self.beta_method == "kalman" and self.beta_hedge == "rolling":
            temp_kf = KalmanState()
            temp_kf.state_mean = np.copy(kf_state.state_mean)
            temp_kf.state_cov = np.copy(kf_state.state_cov)
            precalc_kalman_beta[test_start_pos] = market_beta
            for i in range(test_start_pos + 1, N):
                precalc_kalman_beta[i] = temp_kf.update(Y_vals[i], X_vals[i])

        precalc_win_free = np.full(N, None, dtype=object)
        precalc_hurst_free = np.full(N, 0.5)

        if self.beta_hedge == "no_hedge":
            base_beta_arr = np.ones(N)
        elif self.beta_hedge == "static":
            base_beta_arr = np.full(N, market_beta)
        else:
            base_beta_arr = (
                precalc_kalman_beta
                if self.beta_method == "kalman"
                else precalc_ols_beta
            )

        for i in range(test_start_pos, N):
            b = base_beta_arr[i]
            slice_x = X_vals[i - lookback_len + 1 : i + 1]
            slice_y = Y_vals[i - lookback_len + 1 : i + 1]

            if self.window_method == "rolling":
                precalc_win_free[i] = calculate_half_life_window(
                    slice_x, slice_y, b, self.valid_window
                )
            precalc_hurst_free[i] = calculate_hurst(slice_x, slice_y, b)

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

        if stop_loss is not None and entry_threshold is not None:
            stop_loss_thr = entry_threshold * stop_loss
        else:
            stop_loss_thr = None

        is_bankrupt = False
        results_buffer = []

        for i in range(test_start_pos, len(df)):
            price_x = df[x_col].iloc[i]
            price_y = df[y_col].iloc[i]
            idx = df.index[i]

            if is_bankrupt:
                z_score, spread, mean, std, market_std, hurst = (
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                )
                total_net_pnl = -initial_cash
                equity = 0.0
                drawdown_pct = -1.0
                signal = 0
            else:
                if self.beta_hedge == "rolling" and i != test_start_pos:
                    if self.beta_method == "kalman":
                        market_beta = precalc_kalman_beta[i]
                    elif self.beta_method == "ols" and position_state.position == 0:
                        market_beta = precalc_ols_beta[i]

                if self.window_method == "rolling" and i != test_start_pos:
                    if self.beta_method == "kalman" or (
                        self.beta_method == "ols" and position_state.position == 0
                    ):
                        market_win = precalc_win_free[i]
                    else:
                        slice_x = X_vals[i - lookback_len + 1 : i + 1]
                        slice_y = Y_vals[i - lookback_len + 1 : i + 1]
                        market_win = calculate_half_life_window(
                            slice_x, slice_y, market_beta, self.valid_window
                        )

                if (
                    position_state.position != 0
                    and position_state.entry_win is not None
                ):
                    win = position_state.entry_win
                else:
                    win = market_win

                if (
                    self.time_decay_sl
                    and win is not None
                    and exit_threshold is not None
                ):
                    time_decay_start = 0.5
                    time_decay_end = 1.0

                    hl_diff = (time_decay_end * win) - (time_decay_start * win)
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

                if market_win is None or market_beta <= 0:
                    z_score, spread, mean, std, market_std, hurst = (
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                    )
                else:
                    slice_x_win = X_vals[i - win + 1 : i + 1]
                    slice_y_win = Y_vals[i - win + 1 : i + 1]
                    spread, mean, market_std = calculate_spread_statistics(
                        slice_x_win, slice_y_win, beta
                    )

                    if (
                        position_state.position != 0
                        and position_state.entry_std is not None
                    ):
                        std = position_state.entry_std
                    else:
                        std = market_std

                    z_score = calculate_z_score(spread=spread, mean=mean, std=std)

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
                    and position_state.time_in_pos >= time_decay_start * win
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

                if self.beta_method == "kalman" or position_state.position == 0:
                    hurst = precalc_hurst_free[i]
                else:
                    slice_x = X_vals[i - lookback_len + 1 : i + 1]
                    slice_y = Y_vals[i - lookback_len + 1 : i + 1]
                    hurst = calculate_hurst(slice_x, slice_y, market_beta)

                if self.agent:
                    current_state = AgentState(
                        z_score=z_score,
                        std=market_std,
                        beta=market_beta,
                        hurst=hurst,
                        window=market_win,
                        position=position_state.position,
                        norm_time_in_pos=position_state.time_in_pos / win if win else 0,
                        drawdown_pct=drawdown_pct,
                        current_market_vol=df["market_vol"].iloc[i],
                    )
                    action = self.agent.get_action(current_state)
                else:
                    action, sl_lock = TradeExecutor.decide(
                        position_state=position_state,
                        signal=signal,
                        z_score=z_score,
                        exit_threshold=exit_threshold,
                    )
                    if sl_lock and self.sl_lock:
                        position_state.sl_lock = True

                position_state.open_time = idx

                pnl, fees = TradeExecutor.execute(
                    fee_rate=self.fee_rate,
                    position_state=position_state,
                    action=action,
                    stop_loss_thr=stop_loss_thr,
                    price_x=price_x,
                    price_y=price_y,
                    beta=beta,
                    win=win,
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
                        total_pnl = 0.0

            results_buffer.append(
                {
                    "index": idx,
                    "z_score": z_score,
                    "spread": spread,
                    "mean": mean,
                    "std": position_state.entry_std,
                    "market_std": market_std,
                    "window": position_state.entry_win,
                    "market_win": market_win,
                    "beta": position_state.entry_beta,
                    "market_beta": market_beta,
                    "hurst": hurst,
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
            df = df.iloc[test_start_pos : final_slice_end + 1].copy()
        else:
            df = df.iloc[test_start_pos : end_pos + 1].copy()

        exec_log_df = exec_logger.to_df()
        exec_log_df["ticker"] = self.ticker_x + "-" + self.ticker_y

        return (
            df.drop(
                columns=[
                    source_x_col,
                    source_y_col,
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
        fixed_window: int | None,
        entry_threshold: float | None,
        exit_threshold: float | None,
        stop_loss: float | None,
        test_start: str,
        test_end: str,
        win_test_start: str,
    ) -> StrategyResult:
        """
        Executes the strategy backtest with specific parameters.

        Args:
            fixed_window (int | None): Fixed lookback window size.
            entry_threshold (float | None): Z-score threshold to open a position.
            exit_threshold (float | None): Z-score threshold to close a position.
            stop_loss (float | None): Stop loss multiplier (e.g., 1.05 for 5% from entry_threshold), None if trade without SL.
            test_start (str): Start date for the backtest loop.
            test_end (str): End date for the backtest loop.
            win_test_start (str): Start date for Z-score OU (Half-Life)-based window calculation.

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
            fixed_window=fixed_window,
            win_test_start=win_test_start,
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
