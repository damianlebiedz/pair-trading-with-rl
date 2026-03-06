from itertools import combinations
import pandas as pd
from joblib import Parallel, delayed
from statsmodels.tsa.stattools import coint


def _calculate_coint(x_name, y_name, x_vals, y_vals):
    """
    Calculates the Engle-Granger p-value for a specific asset pair.
    Worker function designed for parallel execution.
    """
    _, p_value, _ = coint(x_vals, y_vals)
    return {"pair": f"{x_name}-{y_name}", "p_value": p_value}


def engle_granger_cointegration(df: pd.DataFrame, n_jobs: int = -1) -> pd.DataFrame:
    """
    Screens all unique asset combinations for cointegration using parallel execution.

    Args:
        df: DataFrame where columns are asset price series.
        n_jobs: Number of CPU cores to use. -1 uses all available processors.

    Returns:
        DataFrame sorted by p-value, identifying the strongest mean-reverting pairs.
    """
    cols = df.columns

    tasks = (
        (cols[i], cols[j], df.iloc[:, i].values, df.iloc[:, j].values)
        for i, j in combinations(range(len(cols)), 2)
    )

    results = Parallel(n_jobs=n_jobs, verbose=1)(
        delayed(_calculate_coint)(*task) for task in tasks
    )

    return (
        pd.DataFrame(results)
        .sort_values(by="p_value", ascending=True)
        .reset_index(drop=True)
    )
