import logging
from typing import Any, Literal
import pandas as pd
import numpy as np

from modules.core.search_methods import random_search
from modules.data_services.merge_utils import aggregate_strategy_results
from modules.performance.objectives import ObjectiveScheme
from modules.performance.stats import calculate_stats
from modules.performance.strategy import Strategy

logger = logging.getLogger(__name__)


class MultiPairOptimizer:
    def __init__(
        self,
        strategies: list[Strategy],
        opt_start: str,
        opt_end: str,
        opt_win_start: int,
        penalty_bad: float,
        n_iter: int,
        interval: str,
        risk_free_rate_annual: float,
        min_trades_per_pair: int,
        initial_cash: float,
        number_of_pairs: int,
    ):
        self.strategies = strategies
        self.opt_start = opt_start
        self.opt_end = opt_end
        self.opt_win_start = opt_win_start
        self.penalty_bad = penalty_bad
        self.n_iter = n_iter
        self.interval = interval
        self.risk_free_rate_annual = risk_free_rate_annual
        self.min_trades_per_pair = min_trades_per_pair
        self.initial_cash = initial_cash
        self.number_of_pairs = number_of_pairs

    def objective(
        self,
        static_params: dict,
        param_dict: dict,
        metric_type: Literal["gross", "net"],
        objective_func: ObjectiveScheme,
    ) -> float:
        """
        Evaluates the aggregate portfolio performance for a specific set of parameters.

        This method simulates the strategy across all configured pairs using the provided
        parameters. It then aggregates the results into a single portfolio equity curve
        (using `aggregate_strategy_results`) to calculate global performance metrics.

        The goal is to find a parameter set that creates a stable portfolio, rather than
        optimizing each pair individually.

        Locking parameters (static_params):
            static_params = {'stop_loss': 1.05}         # 'stop_loss' will be constant 1.05 for all iterations.
            static_params = {'stop_loss': None}         # Trade without 'stop_loss'.

        Args:
            static_params (dict): Constant parameters fixed during this optimization run.
            param_dict (dict): The specific set of variable parameters (e.g., fixed_window,
                thresholds) being tested in this iteration.
            metric_type (Literal["gross", "net"]): Whether to evaluate based on Gross PnL
                or Net PnL (after fees).
            objective_func (ObjectiveScheme): The scoring logic (e.g., Sharpe Ratio)
                used to quantify the portfolio's performance.

        Returns:
            float: The calculated score based on the aggregated portfolio stats.
            Returns `self.penalty_bad` if execution fails, results are empty, or
            scores are invalid (NaN/Inf).
        """
        try:
            params = {**static_params, **param_dict}

            fixed_window = params.get("fixed_window")
            entry_threshold = params.get("entry_threshold")
            exit_threshold = params.get("exit_threshold")
            stop_loss = params.get("stop_loss")

            results = []

            for strat in self.strategies:
                try:
                    res = strat.run_strategy(
                        fixed_window=fixed_window,
                        entry_threshold=entry_threshold,
                        exit_threshold=exit_threshold,
                        stop_loss=stop_loss,
                        test_start=self.opt_start,
                        test_end=self.opt_end,
                        win_test_start=self.opt_win_start,
                    )
                    results.append(res)

                except Exception as e:
                    logger.error(
                        f"[MultiPairOpt Error] during {strat.ticker_x}-{strat.ticker_y}: {e}"
                    )
                    raise e

            merged_df, merged_exec_res = aggregate_strategy_results(
                results=results,
                initial_cash=self.initial_cash,
            )

            stats = calculate_stats(
                df=merged_df,
                exec_log_df=merged_exec_res,
                initial_cash=self.initial_cash,
                interval=self.interval,
                risk_free_rate_annual=self.risk_free_rate_annual,
            )

            score = objective_func.calculate(stats=stats, metric_type=metric_type)

            if isinstance(score, pd.Series):
                score = score.iloc[0]

            if pd.isna(score) or np.isinf(score):
                return self.penalty_bad

            return float(score)

        except Exception as e:
            logger.error(f"[MultiPairOpt Error]: {e}")
            return self.penalty_bad

    def run(
        self,
        static_params: dict[str, Any],
        param_space: list[Any],
        metric_type: Literal["gross", "net"],
        objective_func: ObjectiveScheme,
    ) -> tuple[dict, float]:

        def wrapper_func(**kwargs) -> float:
            return self.objective(
                static_params=static_params,
                param_dict=kwargs,
                metric_type=metric_type,
                objective_func=objective_func,
            )

        best_params, best_score = random_search(
            strategy_func=wrapper_func,
            param_space=param_space,
            static_params=static_params,
            metric_type=metric_type,
            n_iter=self.n_iter,
            penalty_bad=self.penalty_bad,
        )

        return best_params, best_score
