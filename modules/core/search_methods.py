from random import randint, uniform
from typing import Callable, Literal
import numpy as np
from joblib import Parallel, delayed
from skopt.space import Integer, Real


def random_search(
    strategy_func: Callable,
    param_space: list,
    static_params: dict,
    metric_type: Literal["gross", "net"],
    n_iter: int,
    penalty_bad: float,
) -> tuple[dict, float]:
    def evaluate_point(p, idx) -> tuple[float, dict]:
        try:
            val = strategy_func(**{**static_params, **p}, metric_type=metric_type)
            if val is None or np.isnan(val) or val == 0 or np.isinf(val):
                score = penalty_bad
            else:
                score = float(val)
        except Exception as e:
            print(f"[Opt Error] Iter {idx}: {e}")
            score = penalty_bad

        print(f"Iteration {idx + 1}/{n_iter}")
        return score, p

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
