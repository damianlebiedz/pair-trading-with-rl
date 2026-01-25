"""Perform statistical tests for the pair selection."""

from itertools import combinations
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen


def engle_granger_cointegration(df: pd.DataFrame) -> pd.DataFrame:
    df_clean = df[df.columns].dropna()
    results = []

    df_clean = np.log(df_clean)

    for x, y in list(combinations(df.columns, 2)):
        X = df_clean[x].values
        Y = df_clean[y].values
        score, p_value, _ = coint(Y, X)
        results.append(
            {
                "pair": f"{x}-{y}",
                "eg_p_value": p_value,
            }
        )

    return (
        pd.DataFrame(results)
        .sort_values(by="eg_p_value", ascending=True)
        .reset_index(drop=True)
    )


def johansen_cointegration(
    df: pd.DataFrame, det_order: int = 0, k_ar_diff: int = 1
) -> pd.DataFrame:
    df_clean = df[df.columns].dropna()
    results = []

    for x, y in combinations(df.columns, 2):
        data = df_clean[[x, y]]
        result = coint_johansen(data, det_order, k_ar_diff)

        trace_stat = result.lr1[0]
        crit_95 = result.cvt[0, 1]
        crit_99 = result.cvt[0, 2]

        results.append(
            {
                "pair": f"{x}-{y}",
                "trace_stat": trace_stat,
                "crit_95": crit_95,
                "crit_99": crit_99,
                "trace_stat - crit_95": trace_stat - crit_95,
                "trace_stat - crit_99": trace_stat - crit_99,
            }
        )

    return (
        pd.DataFrame(results)
        .sort_values(by="trace_stat - crit_95", ascending=False)
        .reset_index(drop=True)
    )
