from typing import Callable
from skopt import gp_minimize
from skopt.space import Integer, Real
import numpy as np
from random import uniform, randint
from joblib import Parallel, delayed


def random_search(
    strategy_func: Callable,
    param_space: list,
    static_params: dict,
    metric: tuple[str, str],
    n_iter: int = 1000,
    n_jobs: int = -1,
    replicates: int = 1,
    penalty_bad: float = -1e2,
) -> tuple[dict, float]:
    def evaluate_point(pd, idx) -> tuple[float, dict]:
        scores = []
        for _ in range(replicates):
            try:
                val = strategy_func(**{**static_params, **pd}, metric=metric)
                if val is None or np.isnan(val) or val == 0 or np.isinf(val):
                    scores.append(penalty_bad)
                else:
                    scores.append(float(val))
            except Exception as e:
                print(f"[Opt Error] Iter {idx}: {e}")
                scores.append(penalty_bad)

        avg_score = float(np.mean(scores))
        print(f"Iteration {idx + 1}/{n_iter} | Score: {avg_score:.4f}")
        return avg_score, pd

    pdicts = []
    for _ in range(n_iter):
        pdict = {}
        for dim in param_space:
            if isinstance(dim, Integer):
                pdict[dim.name] = randint(dim.low, dim.high)
            elif isinstance(dim, Real):
                pdict[dim.name] = uniform(dim.low, dim.high)
        pdicts.append(pdict)

    results = Parallel(n_jobs=n_jobs, backend="loky")(
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
    n_jobs: int = -1,
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
        n_jobs=n_jobs,
        random_state=random_state,
        verbose=True,
    )

    best_params_values = result.x
    best_score_inverted = result.fun

    best_params = {dim.name: val for dim, val in zip(param_space, best_params_values)}
    best_score = -best_score_inverted

    return best_params, best_score
