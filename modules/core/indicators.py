import numpy as np


def generate_signal(
    z_score: float | None,
    prev_z_score: float | None,
    entry_threshold: float | None,
    stop_loss_thr: float | None,
    delayed_entry: bool,
) -> int:
    """
    Generates trading signals based on Z-Score threshold crossings.

    Signal Logic:
    1. **Long Signal (1)**: Implies spread is too low (Long Spread = Long X / Short Y).
    2. **Short Signal (-1)**: Implies spread is too high (Short Spread = Short X / Long Y).
    3. **Hold (0)**: No new entry signal.

    Entry Modes:
    - **Standard Entry**: Triggered when the Z-Score crosses *out* of the bands
      (e.g., z_score > entry_threshold). Captures divergence immediately.
    - **Delayed Entry**: Triggered when the Z-Score is extreme but crosses *back* towards the mean (e.g., prev > thr and curr < thr). Captures mean reversion
      momentum and avoids "catching a falling knife".

    Stop Loss:
    - If enabled, prevents opening positions if the spread has diverged beyond
      the 'stop_loss_thr'.

    Returns:
        int: Signal direction (1 for Long, -1 for Short, 0 for Neutral).
    """
    if prev_z_score is None or z_score is None or entry_threshold is None:
        return 0

    if delayed_entry:
        long_signal = prev_z_score <= -entry_threshold < z_score
        short_signal = prev_z_score >= entry_threshold > z_score
    else:
        if stop_loss_thr:
            long_signal = -stop_loss_thr <= z_score <= -entry_threshold < prev_z_score
            short_signal = prev_z_score < entry_threshold <= z_score <= stop_loss_thr
        else:
            long_signal = z_score <= -entry_threshold < prev_z_score
            short_signal = prev_z_score < entry_threshold <= z_score

    if long_signal:
        return 1
    elif short_signal:
        return -1
    else:
        return 0


def calculate_beta(
    X_slice: np.ndarray,
    Y_slice: np.ndarray,
) -> float:
    """
    Calculates the hedge ratio (beta) using the Ordinary Least Squares (OLS) method, where 'x_col' is the target
    and 'y_col' is the feature.

    The function determines 'beta' such that the spread is defined as: spread = x - beta * y.
    """
    cov_matrix = np.cov(X_slice, Y_slice, ddof=1)
    var_y = cov_matrix[1, 1]
    if var_y == 0:
        return 0.0
    beta = cov_matrix[0, 1] / var_y
    return beta


def calculate_z_score(spread: float, mean: float, std: float) -> float | None:
    """
    Calculates the Z-Score (standard score) for a specific spread value.

    The Z-Score measures how many standard deviations the current spread is
    from its historical mean. In pairs trading, it is the primary indicator
    used to identify entry and exit signals.

    Returns:
        float | None: The calculated Z-Score value, or None if the standard
            deviation is zero (avoiding division by zero).
    """
    if std == 0:
        return None
    return (spread - mean) / std


def calculate_hurst(
    X_slice: np.ndarray, Y_slice: np.ndarray, beta: float, max_lags: int = 20
) -> float:
    """
    Calculates the Hurst Exponent to determine the time series memory.

    The Hurst exponent (H) characterizes the long-term memory of a time series.
    It is used to identify whether a series is mean-reverting, trending, or
    following a random walk.

    Hurst Exponent Interpretation:
        - H < 0.5: Mean-reverting series (anti-persistent).
        - H = 0.5: Random walk (Geometric Brownian Motion).
        - H > 0.5: Trending series (persistent).
    """
    series_val = X_slice - beta * Y_slice
    if len(series_val) < max_lags * 2:
        return 0.5

    lags = range(2, max_lags)
    tau = [np.std(series_val[lag:] - series_val[:-lag], ddof=0) for lag in lags]

    if not tau:
        return 0.5
    return np.polyfit(np.log(list(lags)), np.log(tau), 1)[0]


def calculate_spread_statistics(
    X_slice: np.ndarray, Y_slice: np.ndarray, beta: float
) -> tuple[float, float, float]:
    """
    Calculates basic spread statistics for a pair of assets.

    This function derives the spread time series based on the formula:
    spread = x - beta * y. It then computes the most recent spread value,
    the rolling mean, and the standard deviation for the provided data window.
    """
    spread_arr = X_slice - beta * Y_slice
    return spread_arr[-1], np.mean(spread_arr), np.std(spread_arr, ddof=1)
