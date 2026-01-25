from typing import Literal
import pandas as pd

from modules.core.execution import TradeExecutor
from modules.core.indicators import (
    calculate_z_score,
    generate_signal,
    calculate_beta,
    calculate_half_life_window,
)
from modules.core.search_methods import random_search
from modules.data_services.data_loaders import load_pair
from modules.data_services.data_utils import add_log_prices
from modules.core.models import PositionState, ExecutionContext, StrategyResult
from modules.performance.stats import calculate_stats


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
        min_trades_per_pair (int): Minimum number of trades per pair for the objective.
        window (str): Window mode: "fixed" (manual size), "rolling" (rolling half-life), "static" (initial half-life).
        beta_hedge (str): Hedge ratio mode: "rolling", "static" or "no_hedge".
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
        min_trades_per_pair: int,
        window: Literal["rolling", "static", "fixed"],
        beta_hedge: Literal["rolling", "static", "no_hedge"],
    ):

        if window not in ["rolling", "static", "fixed"]:
            raise ValueError("Invalid window: should be 'rolling', 'static' or 'fixed'")

        if beta_hedge not in ["rolling", "static", "no_hedge"]:
            raise ValueError(
                "Invalid beta_hedge: should be 'rolling', 'static' or 'no_hedge'"
            )

        self.ticker_x = ticker_x
        self.ticker_y = ticker_y
        self.start = start
        self.end = end
        self.interval = interval
        self.fee_rate = fee_rate
        self.initial_cash = initial_cash
        self.risk_free_rate_annual = risk_free_rate_annual
        self.min_trades_per_pair = min_trades_per_pair
        self.window = window
        self.beta_hedge = beta_hedge

        self.exec_ctx = ExecutionContext(
            ticker_x=self.ticker_x,
            ticker_y=self.ticker_y,
            initial_cash=self.initial_cash,
            fee_rate=self.fee_rate,
        )

        self.data = load_pair(
            x=ticker_x, y=ticker_y, start=start, end=end, interval=interval
        )

        add_log_prices(self.data, self.ticker_x, self.ticker_y)

    def _execute_loop(
        self,
        df: pd.DataFrame,
        entry_threshold: float,
        exit_threshold: float,
        stop_loss: float,
        test_start: str,
        test_end: str,
        window: Literal["rolling", "static", "fixed"],
        window_factor: float | int,
        beta_test_start: str,
    ) -> pd.DataFrame:
        df = df.copy()

        x_col = self.ticker_x
        y_col = self.ticker_y
        source_x_col = f"{x_col}_log"
        source_y_col = f"{y_col}_log"

        beta_hedge = self.beta_hedge

        test_start_pos = df.index.get_loc(pd.to_datetime(test_start))
        start_pos = df.index.get_loc(pd.to_datetime(beta_test_start))
        end_pos = df.index.get_loc(pd.to_datetime(test_end))

        if beta_hedge == "no_hedge":
            initial_beta = 1.0
        else:
            initial_beta = calculate_beta(
                x_col=source_x_col,
                y_col=source_y_col,
                df=df.iloc[start_pos:test_start_pos],
            )

        if window == "fixed":
            initial_win = int(window_factor)
        else:
            initial_win = calculate_half_life_window(
                x_col=source_x_col,
                y_col=source_y_col,
                beta=initial_beta,
                df=df.iloc[start_pos:test_start_pos],
                window_factor=window_factor,
            )

        start_z_score = 0.0
        if (
            initial_win is not None
            and initial_beta > 0
            and 2 <= initial_win <= (test_start_pos - start_pos)
        ):
            start_z_score = calculate_z_score(
                x_col=source_x_col,
                y_col=source_y_col,
                beta=initial_beta,
                df=df.iloc[test_start_pos - initial_win : test_start_pos],
            )
            if pd.isna(start_z_score):
                start_z_score = 0.0

        total_fees = 0.0
        total_pnl = 0.0
        prev_pnl = 0.0
        position_state = PositionState()

        prev_z_score = start_z_score
        beta = initial_beta
        win = initial_win

        results_buffer = []

        for i in range(test_start_pos, len(df)):
            if total_pnl == -self.initial_cash:
                df = df.iloc[:i].copy()
                break

            price_x = df[x_col].iloc[i]
            price_y = df[y_col].iloc[i]

            if beta_hedge == "rolling":
                prev_beta = beta
                beta = calculate_beta(
                    x_col=source_x_col,
                    y_col=source_y_col,
                    df=df.iloc[start_pos + i - test_start_pos : i],
                )
                if beta <= 0:
                    beta = prev_beta

            if window == "rolling":
                prev_win = win
                win = calculate_half_life_window(
                    x_col=source_x_col,
                    y_col=source_y_col,
                    beta=beta,
                    df=df.iloc[start_pos + i - test_start_pos : i],
                    window_factor=window_factor,
                )
                if win is None or win > i or win < 2:
                    win = prev_win

            z_score = calculate_z_score(
                x_col=source_x_col,
                y_col=source_y_col,
                beta=beta,
                df=df.iloc[i - win : i + 1],
            )

            signal = generate_signal(entry_threshold=entry_threshold, z_score=z_score)

            position_state.signal = signal
            idx = df.index[i]

            pnl, total_fees = TradeExecutor.execute(
                ctx=self.exec_ctx,
                position_state=position_state,
                price_x=price_x,
                price_y=price_y,
                z_score=z_score,
                prev_z_score=prev_z_score,
                beta=beta,
                total_fees=total_fees,
                entry_threshold=entry_threshold,
                exit_threshold=exit_threshold,
                stop_loss=stop_loss,
            )

            prev_z_score = 0.0 if z_score is None or pd.isna(z_score) else z_score

            if pnl != 0:
                total_pnl = pnl + prev_pnl
                if (
                    position_state.position != 0
                    and position_state.prev_position != position_state.position
                ):
                    prev_pnl = total_pnl
            else:
                prev_pnl = total_pnl

            if total_pnl <= -self.initial_cash:
                total_pnl = -self.initial_cash

            results_buffer.append(
                {
                    "index": idx,
                    "z_score": z_score,
                    "window": win,
                    "beta": beta,
                    "entry_thr": entry_threshold,
                    "exit_thr": exit_threshold,
                    "sl_thr": position_state.stop_loss_threshold,
                    "q_x": position_state.q_x,
                    "q_y": position_state.q_y,
                    "w_x": position_state.w_x,
                    "w_y": position_state.w_y,
                    "signal": position_state.signal,
                    "position": position_state.position,
                    "total_return": total_pnl,
                    "total_fees": total_fees,
                    "net_return": total_pnl - total_fees,
                }
            )

            position_state.prev_position = position_state.position

        if results_buffer:
            results_df = pd.DataFrame(results_buffer)
            results_df.set_index("index", inplace=True)

            df.loc[results_df.index, results_df.columns] = results_df

        df["total_return_pct"] = df["total_return"] / self.initial_cash
        df["net_return_pct"] = df["net_return"] / self.initial_cash

        if results_buffer:
            last_processed_idx = results_buffer[-1]["index"]
            last_pos_loc = df.index.get_loc(last_processed_idx)
            final_slice_end = min(last_pos_loc, end_pos)
            df = df.iloc[test_start_pos : final_slice_end + 1].copy()
        else:
            df = df.iloc[test_start_pos : end_pos + 1].copy()

        df.iloc[-1, df.columns == "position"] = 0

        return df.drop(columns=[source_x_col, source_y_col], errors="ignore")

    def run_strategy(
        self,
        window_factor: float | int,
        entry_threshold: float,
        exit_threshold: float,
        stop_loss: float,
        test_start: str,
        test_end: str,
        beta_test_start: str,
    ) -> StrategyResult:
        """
        Executes the strategy backtest with specific parameters.

        Args:
            window_factor (float | int): Dual-purpose parameter controlling the lookback window.
                The interpretation depends strictly on the `window` mode defined in `__init__`:
                * If window="fixed":
                    `window_factor` is the exact window size (Integer).
                    Example: `100` means the strategy looks back exactly 100 bars.
                * If window="rolling" or "static":
                    `window_factor` is the Half-Life multiplier (Float).
                    Example: `2.5` means the window size is calculated as `2.5 * Half_Life`.
            entry_threshold (float): Z-score threshold to open a position.
            exit_threshold (float): Z-score threshold to close a position.
            stop_loss (float): Stop loss multiplier (e.g., 1.05 for 5% from current Z-score), None if trade without SL.
            test_start (str): Start date for the backtest loop.
            test_end (str): End date for the backtest loop.
            beta_test_start (str): Start date for beta and Z-score window calculation.

        Returns:
            StrategyResult: Object containing backtest data and performance statistics.
        """

        data = self._execute_loop(
            df=self.data,
            entry_threshold=entry_threshold,
            exit_threshold=exit_threshold,
            stop_loss=stop_loss,
            test_start=test_start,
            test_end=test_end,
            window=self.window,
            window_factor=window_factor,
            beta_test_start=beta_test_start,
        )

        stats = calculate_stats(
            df=data,
            initial_cash=self.initial_cash,
            interval=self.interval,
            risk_free_rate_annual=self.risk_free_rate_annual,
            min_trades_per_pair=self.min_trades_per_pair,
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
        )

    def run_optimization(
        self,
        static_params: dict,
        param_space: list,
        metric: tuple[str, str],
        opt_start: str,
        opt_end: str,
        beta_opt_start: str,
        n_iter: int | None = None,
        replicates: int | None = None,
        penalty_bad: int | None = None,
    ) -> tuple[dict, float]:
        """
        Runs optimization to find the best parameter combination for the strategy.

        Scenario A: Fixed Window Size (window="fixed")
        -> 'window_factor' represents the exact window length (int)
        >>> from skopt.space import Integer, Real
        >>> param_space = [
        >>>     Integer(10, 300, name='window_factor'), # Search window size from 10 to 300
        >>>     Real(1.0, 3.0, name='entry_threshold'),
        >>>     ...
        >>> ]

        Scenario B: Dynamic Window (window="rolling" or "static")
        -> 'window_factor' represents the Half-Life multiplier (float)
        >>> from skopt.space import Real
        >>> param_space = [
        >>>     Real(0.5, 4.0, name='window_factor'),   # Search multiplier from 0.5 to 4.0
        >>>     Real(1.0, 3.0, name='entry_threshold'),
        >>>     ...
        >>> ]

        Scenario C: Locking parameters (static_params)
        >>> static_params = {'stop_loss': 1.05}         # 'stop_loss' will be constant 1.05 for all iterations.
            static_params = {'stop_loss': None}         # Trade without 'stop_loss'.

        Args:
            static_params (dict): Dictionary of parameters to keep constant (not optimized).
            param_space (list): List of skopt Dimensions (Integer/Real) for parameters to optimize.
            metric (tuple[str, str]): Metric to minimize/maximize (e.g., ('stats', 'sharpe_ratio')).
            opt_start (str): Start date for optimization period.
            opt_end (str): End date for optimization period.
            beta_opt_start (str): Start date beta and Z-score window calculation.
            n_iter (int, optional): Number of optimization iterations.
            replicates (int, optional): Number of runs per param set to average results (reduces noise).
            penalty_bad (int, optional): Score assigned to failed/invalid runs.

        Returns:
            tuple[dict, float]: Best parameters found and the corresponding score.
        """

        def objective_wrapper(
            window_factor: float | int,
            entry_threshold: float,
            exit_threshold: float,
            stop_loss: float,
            **_kwargs,
        ) -> float:
            try:
                result = self.run_strategy(
                    window_factor=window_factor,
                    entry_threshold=entry_threshold,
                    exit_threshold=exit_threshold,
                    stop_loss=stop_loss,
                    test_start=opt_start,
                    test_end=opt_end,
                    beta_test_start=beta_opt_start,
                )

                score = result.stats.loc[metric]

                if isinstance(score, pd.Series):
                    score = score.iloc[0]
                if pd.isna(score):
                    return penalty_bad
                return score

            except Exception as e:
                print(f"Error in optimization run: {e}")
                return penalty_bad

        best_params, best_score = random_search(
            strategy_func=objective_wrapper,
            param_space=param_space,
            static_params=static_params,
            metric=metric,
            n_iter=n_iter,
            replicates=replicates,
            penalty_bad=penalty_bad,
        )

        return best_params, best_score
