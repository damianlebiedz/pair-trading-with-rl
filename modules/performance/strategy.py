from typing import Literal
import pandas as pd

from modules.core.execution import TradeExecutor
from modules.core.indicators import calculate_z_score, generate_signal, calculate_beta
from modules.data_services.data_loaders import load_pair
from modules.data_services.data_preparation import (
    add_log_prices,
    add_c_norm_returns,
    add_c_returns,
    add_c_log_returns,
)
from modules.performance.optimization import bayesian_search
from modules.core.models import PositionState, ExecutionContext, StrategyResult
from modules.performance.stats import calculate_stats


class Strategy:
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
        source: Literal["log", "c_returns", "c_log_returns", "c_norm_returns"],
        beta_hedge: Literal["dynamic_hedge", "static_hedge", None],
    ):

        if source not in ["log", "c_returns", "c_log_returns", "c_norm_returns"]:
            raise ValueError(
                "Invalid source: should be 'log', 'c_returns', 'c_log_returns', or 'c_norm_returns'"
            )

        if beta_hedge not in ["dynamic_hedge", "static_hedge", None]:
            raise ValueError(
                "Invalid beta_hedge: should be 'dynamic_hedge', 'static_hedge' or None"
            )

        self.ticker_x = ticker_x
        self.ticker_y = ticker_y
        self.start = start
        self.end = end
        self.interval = interval
        self.fee_rate = fee_rate
        self.initial_cash = initial_cash
        self.risk_free_rate_annual = risk_free_rate_annual
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
        rolling_window: int,
        entry_threshold: float,
        exit_threshold: float,
        stop_loss: float,
        test_start: str,
        test_end: str,
        beta_calculation_start: str | None = None,
        beta_hedge: Literal["dynamic_hedge", "static_hedge", None] | None = None,
    ) -> pd.DataFrame:
        df = df.copy()

        x_col = self.ticker_x
        y_col = self.ticker_y

        total_fees = 0.0
        total_pnl = 0.0
        prev_pnl = 0.0

        position_state = PositionState()

        source_x_col = f"{x_col}_{self.source}"
        source_y_col = f"{y_col}_{self.source}"

        test_start_pos = df.index.get_loc(pd.to_datetime(test_start))
        if test_start_pos - rolling_window < 0:
            raise ValueError("Rolling window cannot be bigger than pre-training period")

        start_pos = None
        if beta_calculation_start is None:
            if beta_hedge in ["static_hedge", "dynamic_hedge"]:
                raise ValueError(
                    "'start' must be provided for 'static_hedge' or 'dynamic_hedge'"
                )
            else:
                pass
        else:
            start_pos = df.index.get_loc(pd.to_datetime(beta_calculation_start))

        beta = 1.0
        if beta_hedge == "static_hedge":
            beta = calculate_beta(
                x_col=source_x_col,
                y_col=source_y_col,
                df=df.iloc[start_pos:test_start_pos],
            )
        elif beta_hedge == "dynamic_hedge":
            pass

        end_pos = df.index.get_loc(pd.to_datetime(test_end))

        df["z_score"] = None

        for i in range(test_start_pos, len(df)):
            if total_pnl == -self.initial_cash:
                df = df.iloc[:i].copy()
                break

            price_x = df[x_col].iloc[i]
            price_y = df[y_col].iloc[i]

            if beta_hedge == "dynamic_hedge":
                beta = calculate_beta(
                    x_col=source_x_col,
                    y_col=source_y_col,
                    df=df.iloc[start_pos + i - test_start_pos : i],
                )

            z_score = calculate_z_score(
                x_col=source_x_col,
                y_col=source_y_col,
                beta=beta,
                df=df.iloc[i - rolling_window : i],
            )

            signal = generate_signal(entry_threshold=entry_threshold, z_score=z_score)

            if beta > 0:
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
            else:
                prev_pnl = total_pnl

            if total_pnl <= -self.initial_cash:
                total_pnl = -self.initial_cash

            df.at[idx, "z_score"] = z_score
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
        rolling_window: int,
        entry_threshold: float,
        exit_threshold: float,
        stop_loss: float,
        test_start: str,
        test_end: str,
        beta_calculation_start: str | None = None,
        beta_hedge: str | None = None,
    ) -> StrategyResult:

        if beta_hedge is None:
            beta_hedge = self.beta_hedge

        data = self._execute_loop(
            df=self.data,
            rolling_window=rolling_window,
            entry_threshold=entry_threshold,
            exit_threshold=exit_threshold,
            stop_loss=stop_loss,
            test_start=test_start,
            test_end=test_end,
            beta_calculation_start=beta_calculation_start,
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
            rolling_window=rolling_window,
            stats=stats,
        )

    def run_optimization(
        self,
        static_params: dict,
        param_space: list,
        metric: tuple[str, str],
        opt_start: str,
        opt_end: str,
        opt_beta_calculation_start: str | None = None,
        n_iter: int | None = None,
        random_state: int | None = None,
        replicates: int | None = None,
        penalty_bad: int | None = None,
    ) -> tuple[dict, float]:

        if self.beta_hedge == "dynamic_hedge":
            beta_hedge = "static_hedge"
        else:
            beta_hedge = None

        def objective_wrapper(
            rolling_window: int,
            entry_threshold: float,
            exit_threshold: float,
            stop_loss: float,
            **_kwargs,
        ) -> float:
            try:
                result = self.run_strategy(
                    rolling_window=int(rolling_window),
                    entry_threshold=entry_threshold,
                    exit_threshold=exit_threshold,
                    stop_loss=stop_loss,
                    test_start=opt_start,
                    test_end=opt_end,
                    beta_calculation_start=opt_beta_calculation_start,
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

        best_params, best_score = bayesian_search(
            strategy_func=objective_wrapper,
            param_space=param_space,
            static_params=static_params,
            metric=metric,
            n_iter=n_iter,
            random_state=random_state,
            replicates=replicates,
            penalty_bad=penalty_bad,
        )

        return best_params, best_score
