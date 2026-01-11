"""Perform statistical tests for the pair selection."""

from itertools import combinations
from typing import Literal

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen


def ssd_cumulative_returns(df: pd.DataFrame) -> pd.DataFrame:
    df_temp = df.copy()

    for col in df_temp.columns:
        df_temp[col] = df_temp[col] / df_temp[col].iloc[0]

    results = []
    for x, y in combinations(df_temp.columns, 2):
        ssd = np.sum((df_temp[x] - df_temp[y]) ** 2)
        results.append({"pair": f"{x}-{y}", "ssd": ssd})

    results_df = pd.DataFrame(results)

    return results_df.sort_values(by="ssd", ascending=True).reset_index(drop=True)


def pearson_correlation(
    df: pd.DataFrame, source: Literal["prices", "returns", "log_returns"]
) -> pd.DataFrame:
    df_clean = df[df.columns].dropna()

    if source == "returns":
        df_clean = df_clean.pct_change().dropna()
    elif source == "log_returns":
        df_clean = (df_clean / df_clean.shift(1)).apply(np.log).dropna()
    elif source != "prices":
        raise ValueError("'source' must be 'prices', 'returns' or 'log_returns'")

    corr_matrix = df_clean.corr(method="pearson")
    corr_df = (
        corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        .stack()
        .reset_index()
    )
    corr_df.columns = ["x", "y", f"corr_{source}"]
    corr_df["pair"] = corr_df["x"].astype(str) + "-" + corr_df["y"].astype(str)
    corr_df = corr_df[["pair", f"corr_{source}"]]

    return corr_df.sort_values(by=f"corr_{source}", ascending=False).reset_index(
        drop=True
    )


def engle_granger_cointegration(
    df: pd.DataFrame, source: Literal["prices", "log_prices"]
) -> pd.DataFrame:
    df_clean = df[df.columns].dropna()
    results = []

    if source == "log_prices":
        df_clean = np.log(df_clean)
    elif source != "prices":
        raise ValueError("'source' must be 'prices' or 'log_prices'")

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
