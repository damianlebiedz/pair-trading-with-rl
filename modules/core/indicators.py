import numpy as np
import pandas as pd
import statsmodels.api as sm


def generate_signal(entry_threshold: float, z_score: float) -> int:
    """
    Generate signals for trades depends on current z-score.

    Signal = 1:     Long X, Short Y
    Signal = -1:    Short X, Long Y
    Signal = 0:     do nothing
    """
    signal = 0
    if z_score is not None:
        if z_score <= -entry_threshold:
            signal = 1
        elif z_score >= entry_threshold:
            signal = -1

    return signal


def calculate_beta(x_col: str, y_col: str, df: pd.DataFrame) -> float:
    """Calculate beta from OLS."""
    X = sm.add_constant(df[y_col])
    y = df[x_col]
    model = sm.OLS(y, X, missing="drop").fit()
    beta = model.params[y_col]

    return beta


def calculate_z_score(
    x_col: str, y_col: str, beta: float, df: pd.DataFrame
) -> float | None:
    """Calculate z-score with provided beta."""
    spread_series = df[x_col] - (beta * df[y_col])

    historical = spread_series.iloc[:-1]
    spread = spread_series.iloc[-1]

    mean = historical.mean()
    std = historical.std()

    if std == 0:
        return None
    z_score = (spread - mean) / std

    return z_score


def calculate_half_life_window(
    x_col: str,
    y_col: str,
    beta: float,
    df: pd.DataFrame,
    window_factor: float,
) -> int | None:
    """
    Estimate OU process half-life for a spread and derive a rolling window size.

    Returns:
        int: window size based on half-life * window_factor
        None: if spread is not mean-reverting or window is invalid
    """
    # Construct spread using pre-estimated hedge ratio
    series = df[x_col] - (beta * df[y_col])

    # OU process discretization: ΔX_t = λ X_{t-1} + ε_t
    lag = series.shift(1)
    ret = series - lag

    # Align time series
    lag = lag.iloc[1:]
    ret = ret.iloc[1:]

    # Regress spread changes on lagged level to estimate mean reversion speed (λ)
    X = sm.add_constant(lag)
    model = sm.OLS(ret, X, missing="drop").fit()
    lam = model.params.iloc[1]

    # Reject non-mean-reverting spreads (λ >= 0)
    if lam >= 0:
        return None

    # OU half-life
    half_life = -np.log(2) / lam
    window = int(half_life * window_factor)

    # Reject windows larger than available data
    if window > len(df):
        return None

    return window
