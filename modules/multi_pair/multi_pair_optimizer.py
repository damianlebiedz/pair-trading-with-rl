import logging
from typing import Any
import pandas as pd
from omegaconf import DictConfig
import numpy as np

from modules.core.search_methods import random_search
from modules.multi_pair.multi_pair_utils import aggregate_strategy_results
from modules.performance.stats import calculate_multi_pair_stats
from modules.performance.strategy import Strategy

logger = logging.getLogger(__name__)


class MultiPairOptimizer:
    def __init__(self, strategies: list[Strategy], cfg: DictConfig):
        self.strategies = strategies
        self.cfg = cfg
        self.opt_start = cfg.performance.optimization.start
        self.opt_end = cfg.performance.optimization.end
        self.beta_opt_start = cfg.performance.optimization.beta_start
        self.penalty_bad = cfg.performance.optimization.penalty_bad
        self.interval = strategies[0].interval
        self.risk_free_rate_annual = strategies[0].risk_free_rate_annual
        self.min_trades_per_pair = cfg.performance.optimization.min_trades_per_pair

        self.total_initial_cash = sum(s.initial_cash for s in strategies)
        self.number_of_pairs = len(strategies)

    def objective(
        self, static_params: dict, param_dict: dict, metric: tuple[str, str]
    ) -> float:
        """Calculates the objective score for a given set of parameters across all pairs."""
        try:
            params = {**static_params, **param_dict}

            window_factor = params.get("window_factor")
            entry_threshold = params.get("entry_threshold")
            exit_threshold = params.get("exit_threshold")
            stop_loss = params.get("stop_loss")

            results = []
            individual_stats_dfs = []

            for strat in self.strategies:
                res = strat.run_strategy(
                    window_factor=window_factor,
                    entry_threshold=entry_threshold,
                    exit_threshold=exit_threshold,
                    stop_loss=stop_loss,
                    test_start=self.opt_start,
                    test_end=self.opt_end,
                    beta_test_start=self.beta_opt_start,
                )
                results.append(res)
                individual_stats_dfs.append(res.stats)

            merged_df = aggregate_strategy_results(results, self.total_initial_cash)

            stats = calculate_multi_pair_stats(
                merged_df=merged_df,
                individual_stats_dfs=individual_stats_dfs,
                total_initial_cash=self.total_initial_cash,
                interval=self.interval,
                risk_free_rate_annual=self.risk_free_rate_annual,
                number_of_pairs=self.number_of_pairs,
                min_trades_per_pair=self.min_trades_per_pair,
            )

            score = stats.loc[metric]
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
        metric: tuple[str, str],
    ) -> tuple[dict, float]:

        def wrapper_func(**kwargs) -> float:
            metric_arg = kwargs.pop("metric", metric)

            return self.objective(
                static_params={}, param_dict=kwargs, metric=metric_arg
            )

        best_params, best_score = random_search(
            strategy_func=wrapper_func,
            param_space=param_space,
            static_params=static_params,
            metric=metric,
            n_iter=self.cfg.performance.optimization.n_iter,
            replicates=self.cfg.performance.optimization.replicates,
            penalty_bad=self.penalty_bad,
        )

        return best_params, best_score
