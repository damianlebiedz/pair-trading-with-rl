"""Perform statistical tests for the pair selection."""

from itertools import combinations
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint


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
