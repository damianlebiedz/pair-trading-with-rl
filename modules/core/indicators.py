from typing import Literal
import numpy as np


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
    X_slice: np.ndarray,
    Y_slice: np.ndarray,
    beta_method: Literal["ols", "kalman"],
) -> float:
    """
    Calculates the hedge ratio (beta) using the specified statistical method.

    The function determines 'beta' such that the spread is defined as: spread = x - beta * y.

    Algorithm by method:
    1. **OLS**: Performs a static linear regression (Ordinary Least Squares) where 'x_col' is the target
       and 'y_col' is the feature.
    2. **Kalman**: Applies an online Kalman Filter to estimate the evolving beta step-by-step,
       treating 'y_col' as the observable state predictor for 'x_col'.
    """
    if beta_method not in ["ols", "kalman"]:
        raise ValueError("coint_method should be 'ols' or 'kalman'")

    if beta_method == "ols":
        cov_matrix = np.cov(X_slice, Y_slice, ddof=1)
        var_y = cov_matrix[1, 1]

        if var_y == 0:
            return 0.0

        beta = cov_matrix[0, 1] / var_y
        return beta

    else:
        kf = KalmanState()
        current_beta = 0.0

        for i in range(len(X_slice)):
            current_beta = kf.update(obs_x=Y_slice[i], obs_y=X_slice[i])

        return current_beta


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
    X_slice: np.ndarray, Y_slice: np.ndarray, beta: float, valid_window: tuple[int, int]
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
    """
    if beta <= 0:
        return None

    series = X_slice - beta * Y_slice

    lag = series[:-1]
    diff = series[1:] - series[:-1]

    cov_matrix = np.cov(lag, diff, ddof=1)
    var_lag = cov_matrix[0, 0]
    cov_lag_diff = cov_matrix[0, 1]

    if var_lag == 0:
        return None

    lam = cov_lag_diff / var_lag

    if lam >= 0:
        return None

    half_life = -np.log(2) / lam

    if half_life < valid_window[0] or half_life > valid_window[1]:
        return None

    return int(half_life)


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


def calculate_spread_statistics(X_slice: np.ndarray, Y_slice: np.ndarray, beta: float):
    """
    Calculates basic spread statistics for a pair of assets.

    This function derives the spread time series based on the formula:
    spread = x - beta * y. It then computes the most recent spread value,
    the rolling mean, and the standard deviation for the provided data window.
    """
    spread_arr = X_slice - beta * Y_slice
    return spread_arr[-1], np.mean(spread_arr), np.std(spread_arr, ddof=1)
