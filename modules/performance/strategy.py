from typing import Literal
import pandas as pd

from modules.core.execution import TradeExecutor
from modules.core.indicators import (
    calculate_z_score,
    generate_signal,
    calculate_beta,
    calculate_half_life_window,
)
from modules.data_services.data_loaders import load_pair
from modules.data_services.data_preparation import (
    add_log_prices,
    add_c_norm_returns,
    add_c_returns,
    add_c_log_returns,
)
from modules.performance.optimization import random_search
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
        window (str): Window mode: "fixed" (manual size), "rolling" (dynamic half-life), "static" (initial half-life).
        source (str): Data source type for beta or/and Z-score calculation.
        beta_hedge (str, optional): Hedge ratio mode: "dynamic", "static" or None.
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
        window: Literal["rolling", "static", "fixed"],
        source: Literal["log", "c_returns", "c_log_returns", "c_norm_returns"],
        beta_hedge: Literal["dynamic", "static", None],
    ):

        if window not in ["rolling", "static", "fixed"]:
            raise ValueError("Invalid window: should be 'rolling', 'static' or 'fixed'")

        if source not in ["log", "c_returns", "c_log_returns", "c_norm_returns"]:
            raise ValueError(
                "Invalid source: should be 'log', 'c_returns', 'c_log_returns', or 'c_norm_returns'"
            )

        if beta_hedge not in ["dynamic", "static", None]:
            raise ValueError(
                "Invalid beta_hedge: should be 'dynamic', 'static' or None"
            )

        self.ticker_x = ticker_x
        self.ticker_y = ticker_y
        self.start = start
        self.end = end
        self.interval = interval
        self.fee_rate = fee_rate
        self.initial_cash = initial_cash
        self.risk_free_rate_annual = risk_free_rate_annual
        self.window = window
        self.source = source
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

        source_map = {
            "c_norm_returns": add_c_norm_returns,
            "c_returns": add_c_returns,
            "c_log_returns": add_c_log_returns,
            "log": add_log_prices,
        }
        func_to_call = source_map[self.source]
        func_to_call(self.data, self.ticker_x, self.ticker_y)

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
        beta_hedge: Literal["dynamic", "static", None] | None = None,
    ) -> pd.DataFrame:
        df = df.copy()

        x_col = self.ticker_x
        y_col = self.ticker_y
        source_x_col = f"{x_col}_{self.source}"
        source_y_col = f"{y_col}_{self.source}"

        total_fees = 0.0
        total_pnl = 0.0
        prev_pnl = 0.0
        position_state = PositionState()

        test_start_pos = df.index.get_loc(pd.to_datetime(test_start))
        start_pos = df.index.get_loc(pd.to_datetime(beta_test_start))

        beta = 1.0
        if beta_hedge == "static":
            beta = calculate_beta(
                x_col=source_x_col,
                y_col=source_y_col,
                df=df.iloc[start_pos:test_start_pos],
            )

        win = 0
        if window == "fixed":
            win = int(window_factor)
        elif window == "static":
            win = calculate_half_life_window(
                x_col=source_x_col,
                y_col=source_y_col,
                beta=beta,
                df=df.iloc[start_pos:test_start_pos],
                window_factor=window_factor,
            )

        end_pos = df.index.get_loc(pd.to_datetime(test_end))
        df["z_score"] = None

        for i in range(test_start_pos, len(df)):
            if total_pnl == -self.initial_cash:
                df = df.iloc[:i].copy()
                break

            price_x = df[x_col].iloc[i]
            price_y = df[y_col].iloc[i]

            if beta_hedge == "dynamic":
                beta = calculate_beta(
                    x_col=source_x_col,
                    y_col=source_y_col,
                    df=df.iloc[start_pos + i - test_start_pos : i],
                )

            if window == "rolling":
                win = calculate_half_life_window(
                    x_col=source_x_col,
                    y_col=source_y_col,
                    beta=beta,
                    df=df.iloc[start_pos + i - test_start_pos : i],
                    window_factor=window_factor,
                )

            if win is not None and beta > 0 and 2 <= win <= i:
                z_score = calculate_z_score(
                    x_col=source_x_col,
                    y_col=source_y_col,
                    beta=beta,
                    df=df.iloc[i - win : i + 1],
                )
                signal = generate_signal(
                    entry_threshold=entry_threshold, z_score=z_score
                )
            else:
                z_score = None
                signal = 0

            position_state.signal = signal
            idx = df.index[i]
            prev_z_score = (
                0.0 if pd.isna(df.iloc[i - 1]["z_score"]) else df.iloc[i - 1]["z_score"]
            )

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

            df.at[idx, "z_score"] = z_score
            df.at[idx, "window"] = win
            df.at[idx, "beta"] = beta
            df.at[idx, "entry_thr"] = entry_threshold
            df.at[idx, "exit_thr"] = exit_threshold
            df.at[idx, "sl_thr"] = position_state.stop_loss_threshold
            df.at[idx, "q_x"] = position_state.q_x
            df.at[idx, "q_y"] = position_state.q_y
            df.at[idx, "w_x"] = position_state.w_x
            df.at[idx, "w_y"] = position_state.w_y
            df.at[idx, "signal"] = position_state.signal
            df.at[idx, "position"] = position_state.position
            df.at[idx, "total_return"] = total_pnl
            df.at[idx, "total_fees"] = total_fees
            df.at[idx, "net_return"] = total_pnl - total_fees

            position_state.prev_position = position_state.position

        df["total_return_pct"] = df["total_return"] / self.initial_cash
        df["net_return_pct"] = df["net_return"] / self.initial_cash

        df = df.iloc[test_start_pos : end_pos + 1].copy()

        return df.drop(columns=[source_x_col, source_y_col])

    def run_strategy(
        self,
        window_factor: float | int,
        entry_threshold: float,
        exit_threshold: float,
        stop_loss: float,
        test_start: str,
        test_end: str,
        beta_test_start: str,
        beta_hedge: str | None = None,
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
            stop_loss (float): Stop loss multiplier (e.g., 1.05 for 5% from current Z-score).
            test_start (str): Start date for the backtest loop.
            test_end (str): End date for the backtest loop.
            beta_test_start (str): Start date for beta and Z-score window calculation.
            beta_hedge (str, optional): Override for beta_hedge mode.

        Returns:
            StrategyResult: Object containing backtest data and performance statistics.
        """
        if beta_hedge is None:
            beta_hedge = self.beta_hedge

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
            beta_hedge=beta_hedge,
        )

        stats = calculate_stats(
            df=data,
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
            window_factor=window_factor,
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

        if self.beta_hedge == "dynamic":
            beta_hedge = "static"
        else:
            beta_hedge = None

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
                    beta_hedge=beta_hedge,
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
