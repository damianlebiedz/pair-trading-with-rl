import logging
from typing import Callable, Any
import pandas as pd
from omegaconf import DictConfig
from skopt import gp_minimize
from skopt.space import Integer, Real
import numpy as np
from random import uniform, randint
from joblib import Parallel, delayed

from modules.performance.stats import aggregate_strategy_results, calculate_multi_pair_stats
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

        self.total_initial_cash = sum(s.initial_cash for s in strategies)
        self.number_of_pairs = len(strategies)

    def objective(
            self,
            static_params: dict,
            param_dict: dict,
            metric: tuple[str, str]
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
                beta_hedge_mode = "static" if strat.beta_hedge == "dynamic" else None

                res = strat.run_strategy(
                    window_factor=window_factor,
                    entry_threshold=entry_threshold,
                    exit_threshold=exit_threshold,
                    stop_loss=stop_loss,
                    test_start=self.opt_start,
                    test_end=self.opt_end,
                    beta_test_start=self.beta_opt_start,
                    beta_hedge=beta_hedge_mode
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
                number_of_pairs=self.number_of_pairs
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
            metric: tuple[str, str]
    ) -> tuple[dict, float]:

        def wrapper_func(**kwargs) -> float:
            metric_arg = kwargs.pop("metric", metric)

            return self.objective(static_params={}, param_dict=kwargs, metric=metric_arg)

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


def random_search(
    strategy_func: Callable,
    param_space: list,
    static_params: dict,
    metric: tuple[str, str],
    n_iter: int = 1000,
    replicates: int = 1,
    penalty_bad: float = -100,
) -> tuple[dict, float]:
    def evaluate_point(p, idx) -> tuple[float, dict]:
        scores = []
        for _ in range(replicates):
            try:
                val = strategy_func(**{**static_params, **p}, metric=metric)
                if val is None or np.isnan(val) or val == 0 or np.isinf(val):
                    scores.append(penalty_bad)
                else:
                    scores.append(float(val))
            except Exception as e:
                print(f"[Opt Error] Iter {idx}: {e}")
                scores.append(penalty_bad)

        avg_score = float(np.mean(scores))
        print(f"Iteration {idx + 1}/{n_iter}")
        return avg_score, p

    pdicts = []
    for _ in range(n_iter):
        pdict = {}
        for dim in param_space:
            if isinstance(dim, Integer):
                pdict[dim.name] = randint(dim.low, dim.high)
            elif isinstance(dim, Real):
                pdict[dim.name] = uniform(dim.low, dim.high)
        pdicts.append(pdict)

    results = Parallel(n_jobs=-1, backend="loky")(
        delayed(evaluate_point)(p, i) for i, p in enumerate(pdicts)
    )

    best_score, best_params = max(results, key=lambda x: x[0])
    return best_params, best_score


def bayesian_search(
    strategy_func: Callable,
    param_space: list,
    static_params: dict,
    metric: tuple[str, str],
    n_iter: int = 50,
    random_state: int = 42,
    replicates: int = 1,
    penalty_bad: int = -100,
) -> tuple[dict, float]:
    def objective(params_values):
        pdict = {dim.name: val for dim, val in zip(param_space, params_values)}

        scores = []
        for _ in range(replicates):
            try:
                val = strategy_func(**{**static_params, **pdict}, metric=metric)
                if val is None or np.isnan(val) or val == 0 or np.isinf(val):
                    scores.append(penalty_bad)
                else:
                    scores.append(float(val))
            except Exception as e:
                print(f"[Opt Error] Params {pdict}: {e}")
                scores.append(penalty_bad)

        avg_score = float(np.mean(scores))

        return -avg_score

    result = gp_minimize(
        func=objective,
        dimensions=param_space,
        n_calls=n_iter,
        random_state=random_state,
        verbose=True,
    )

    best_params_values = result.x
    best_score_inverted = result.fun

    best_params = {dim.name: val for dim, val in zip(param_space, best_params_values)}
    best_score = -best_score_inverted

    return best_params, best_score
