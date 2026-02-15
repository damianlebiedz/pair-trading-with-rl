"""Perform statistical tests for the pair selection."""

from itertools import combinations
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen


def engle_granger_cointegration(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pair selection using Engle-Granger cointegration test.
    Returns a DataFrame sorted by p_value (strongest cointegration first).
    """
    df_clean = df[df.columns].dropna()
    results = []

    df_clean = np.log(df_clean)

    for x, y in list(combinations(df.columns, 2)):
        x_vals = df_clean[x].values
        y_vals = df_clean[y].values

        score, p_value, _ = coint(x_vals, y_vals)

        results.append(
            {
                "pair": f"{x}-{y}",
                "p_value": p_value,
            }
        )

    return (
        pd.DataFrame(results)
        .sort_values(by="p_value", ascending=True)
        .reset_index(drop=True)
    )


def johansen_cointegration(
    df: pd.DataFrame, det_order: int = 0, k_ar_diff: int = 1
) -> pd.DataFrame:
    """
    Pair selection using Johansen cointegration test.
    Returns a DataFrame sorted by trace statistic (strongest cointegration first).
    """
    df_clean = df.dropna()
    df_clean = df_clean[df_clean > 0]
    df_clean = df_clean.apply(np.log)

    results = []

    for x, y in combinations(df_clean.columns, 2):
        data = df_clean[[x, y]].to_numpy(dtype=np.float64)
        johansen_res = coint_johansen(data, det_order=det_order, k_ar_diff=k_ar_diff)

        trace_stat = johansen_res.lr1[0]
        max_eig_stat = johansen_res.lr2[0]

        crit_95 = johansen_res.cvt[0, 1]
        crit_99 = johansen_res.cvt[0, 2]

        results.append(
            {
                "pair": f"{x}-{y}",
                "trace_stat": trace_stat,
                "max_eig_stat": max_eig_stat,
                "crit_95": crit_95,
                "crit_99": crit_99,
            }
        )

    return (
        pd.DataFrame(results)
        .sort_values(by="trace_stat", ascending=False)
        .reset_index(drop=True)
    )
