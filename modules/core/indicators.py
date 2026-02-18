from typing import Literal
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.vector_ar.vecm import coint_johansen


class KalmanState:
    """
    Implements an online Kalman Filter for recursive least squares estimation
    of a linear regression model: y = beta * x + alpha.

    The filter treats the regression coefficients (slope 'beta' and intercept 'alpha')
     as the hidden state vector, evolving as a random walk.

    Attributes:
        state_mean (np.ndarray): Current estimate of [beta, alpha].
        state_cov (np.ndarray): Covariance matrix of the state estimate.
        Q (np.ndarray): Process noise covariance matrix (allows adaptation).
        R (float): Measurement noise variance.
    """

    def __init__(self, delta: float = 1e-4, R: float = 1e-3):
        """
        Args:
            delta: Ridge factor for process noise Q. Controls the flexibility of the
                   moving beta (higher = more adaptive/noisy, lower = smoother/stable).
            R: Measurement noise variance. Represents the expected error in the
               observation equation.
        """
        self.state_mean = np.zeros(2)
        self.state_cov = np.ones((2, 2))
        self.Q = np.eye(2) * delta
        self.R = R

    def update(self, obs_x: float, obs_y: float) -> float:
        """
        Updates the filter with a new observation pair and returns the updated beta.

        The observation equation is modeled as: obs_y = beta * obs_x + alpha + noise.

        Args:
            obs_x: Independent variable (Predictor, e.g., Asset Y / Hedge).
            obs_y: Dependent variable (Target, e.g., Asset X).

        Returns:
            float: The updated estimate of the slope coefficient (beta).
        """
        prediction_cov = self.state_cov + self.Q

        H = np.array([obs_x, 1.0])

        y_pred = H.dot(self.state_mean)
        y_residual = obs_y - y_pred

        S = H.dot(prediction_cov).dot(H.T) + self.R
        K = prediction_cov.dot(H.T) / S

        self.state_mean = self.state_mean + K * y_residual
        self.state_cov = (np.eye(2) - np.outer(K, H)).dot(prediction_cov)

        return self.state_mean[0]


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

    Args:
        z_score: Current Z-Score value.
        prev_z_score: Z-Score value from the previous step.
        entry_threshold: Absolute Z-Score level required to consider a trade.
        stop_loss_thr: Absolute Z-Score level where trading is forbidden (risk control).
        delayed_entry: Boolean flag to switch between Standard and Delayed logic.

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
    x_col: str,
    y_col: str,
    df: pd.DataFrame,
    beta_method: Literal["ols", "johansen", "kalman"],
) -> float:
    """
    Calculates the hedge ratio (beta) using the specified statistical method.

    The function determines 'beta' such that the spread is defined as: spread = x - beta * y.

    Algorithm by method:
    1. **OLS**: Performs a static linear regression (Ordinary Least Squares) where 'x_col' is the target
       and 'y_col' is the feature.
    2. **Johansen**: Computes cointegration vectors. Uses the first eigenvector (corresponding to the
       largest eigenvalue) and normalizes it with respect to x to derive beta.
    3. **Kalman**: Applies an online Kalman Filter to estimate the evolving beta step-by-step,
       treating 'y_col' as the observable state predictor for 'x_col'.

    Args:
        x_col: Column name for the dependent asset (X).
        y_col: Column name for the independent asset (Y, hedge).
        df: DataFrame containing the price series.
        beta_method: Method to use ('ols', 'johansen', or 'kalman').

    Returns:
        float: Calculated beta coefficient.

    Raises:
        ValueError: If an invalid beta_method is provided.
    """
    if beta_method not in ["ols", "johansen", "kalman"]:
        raise ValueError("coint_method should be 'ols', 'johansen', or 'kalman'")

    if beta_method == "ols":
        X = sm.add_constant(df[y_col])
        y = df[x_col]
        model = sm.OLS(y, X, missing="drop").fit()
        beta = model.params[y_col]

        return beta

    elif beta_method == "johansen":
        data = df[[x_col, y_col]].dropna()

        johansen_res = coint_johansen(
            data.values,
            det_order=0,
            k_ar_diff=1,
        )

        vec = johansen_res.evec[:, 0]
        beta = -vec[1] / vec[0]

        return beta

    else:
        data = df[[x_col, y_col]].dropna()
        kf = KalmanState()
        current_beta = 0.0

        for i in range(len(data)):
            obs_x = data[y_col].iloc[i]
            obs_y = data[x_col].iloc[i]
            current_beta = kf.update(obs_x, obs_y)

        return current_beta


def calculate_spread_statistics(
    x_col: str, y_col: str, beta: float, df: pd.DataFrame
) -> tuple[float, float, float]:
    """
    Calculates basic spread statistics for a pair of assets.

    This function derives the spread time series based on the formula:
    spread = x - beta * y. It then computes the most recent spread value,
    the rolling mean, and the standard deviation for the provided data window.

    Args:
        x_col (str): Column name for the dependent asset (X).
        y_col (str): Column name for the independent asset (Y, hedge).
        beta (float): The hedge ratio (beta) used to construct the spread.
        df (pd.DataFrame): DataFrame containing price series for both assets.

    Returns:
        tuple[float, float, float]: A tuple containing:
            - spread (float): The current (last) value of the spread series.
            - mean (float): The arithmetic mean of the spread in the current window.
            - std (float): The standard deviation of the spread in the current window.
    """
    spread_series = df[x_col] - (beta * df[y_col])

    spread = spread_series.iloc[-1]
    mean = spread_series.mean()
    std = spread_series.std()

    return spread, mean, std


def calculate_hurst(
    x_col: str, y_col: str, beta: float, df: pd.DataFrame, max_lags: int = 20
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

    Args:
        x_col (str): Column name for asset X.
        y_col (str): Column name for asset Y.
        beta (float): The hedge ratio used to construct the spread.
        df (pd.DataFrame): DataFrame containing price series.
        max_lags (int, optional): The maximum number of lags to consider
            for the calculation. Defaults to 20.

    Returns:
        float: The calculated Hurst Exponent. Returns 0.5 as a fallback if
            there is insufficient data or an error occurs.
    """
    lags = range(2, max_lags)
    tau = []

    spread_series = df[x_col] - (beta * df[y_col])
    series_val = spread_series.values

    if len(series_val) < max_lags * 2:
        return 0.5

    for lag in lags:
        diff = series_val[lag:] - series_val[:-lag]
        if len(diff) == 0:
            continue
        tau.append(np.std(diff))

    if not tau:
        return 0.5

    m = np.polyfit(np.log(lags), np.log(tau), 1)
    hurst = m[0]

    return hurst


def calculate_z_score(spread: float, mean: float, std: float) -> float | None:
    """
    Calculates the Z-Score (standard score) for a specific spread value.

    The Z-Score measures how many standard deviations the current spread is
    from its historical mean. In pairs trading, it is the primary indicator
    used to identify entry and exit signals.

    Args:
        spread (float): The current value of the spread.
        mean (float): The mean value of the spread over the lookback period.
        std (float): The standard deviation of the spread over the lookback period.

    Returns:
        float | None: The calculated Z-Score value, or None if the standard
            deviation is zero (avoiding division by zero).
    """
    if std == 0:
        return None
    return (spread - mean) / std


def calculate_half_life_window(
    x_col: str,
    y_col: str,
    beta: float,
    df: pd.DataFrame,
    valid_window: tuple[int, int],
) -> int | None:
    """
    Estimates the mean-reversion Half-Life via the Ornstein-Uhlenbeck (OU) process
    to derive a dynamic lookback window.

    Methodology:
    1. Construct the spread series using the provided beta: spread = x - beta * y.
    2. Discretize the OU process as: Δspread_t = λ * spread_{t-1} + ε_t.
    3. Regress the daily change in spread (Δspread) against the lagged spread to estimate
       the mean reversion speed (λ).
    4. Validate λ: If λ >= 0, the process is not mean-reverting (explosive or random walk),
       and the function returns None.
    5. Calculate Half-Life: -ln(2) / λ.

    Args:
        x_col: Column name for asset X.
        y_col: Column name for asset Y.
        beta: Hedge ratio.
        df: DataFrame containing price data.
        valid_window: min and max values of window.

    Returns:
        int | None: The calculated window size, or None if the spread is not mean-reverting
        (beta <= 0 or lambda >= 0) or if the calculated window is invalid.
    """
    if beta <= 0:
        return None

    series = df[x_col] - (beta * df[y_col])

    lag = series.shift(1)
    ret = series - lag

    lag = lag.iloc[1:]
    ret = ret.iloc[1:]

    X = sm.add_constant(lag)
    model = sm.OLS(ret, X, missing="drop").fit()
    lam = model.params.iloc[1]

    if lam >= 0:
        return None

    half_life = -np.log(2) / lam

    if half_life < valid_window[0] or half_life > valid_window[1]:
        return None

    return int(half_life)
