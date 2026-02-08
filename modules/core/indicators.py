from typing import Literal, Union, Any
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.vector_ar.vecm import coint_johansen


class KalmanState:
    def __init__(self, delta=1e-4, R=1e-3):
        self.state_mean = np.zeros(2)
        self.state_cov = np.ones((2, 2))
        self.Q = np.eye(2) * delta
        self.R = R

    def update(self, x, y):
        prediction_cov = self.state_cov + self.Q

        H = np.array([x, 1.0])

        y_pred = H.dot(self.state_mean)
        y_residual = y - y_pred

        S = H.dot(prediction_cov).dot(H.T) + self.R
        K = prediction_cov.dot(H.T) / S

        self.state_mean = self.state_mean + K * y_residual
        self.state_cov = (np.eye(2) - np.outer(K, H)).dot(prediction_cov)

        return self.state_mean[0]


def calculate_beta(
    x_col: str,
    y_col: str,
    df: pd.DataFrame,
    beta_method: Literal["ols", "johansen", "kalman"],
) -> Union[float, tuple[float, Any]]:
    """
    Calculate hedge ratio beta using OLS, Johansen or Kalman.
    Returns beta such that spread = x - beta * y
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

        # first cointegrating vector
        vec = johansen_res.evec[:, 0]

        # normalize: x - beta * y
        beta = -vec[1] / vec[0]

        return beta

    else:
        data = df[[x_col, y_col]].dropna()
        kf = KalmanState()
        current_beta = 0.0

        for i in range(len(data)):
            px = data[y_col].iloc[i]
            py = data[x_col].iloc[i]
            current_beta = kf.update(px, py)

        return current_beta


def calculate_z_score(
    x_col: str, y_col: str, beta: float, df: pd.DataFrame
) -> tuple[float | None, float, float, float]:
    """Calculate z-score with provided beta."""
    spread_series = df[x_col] - (beta * df[y_col])

    historical = spread_series.iloc[:-1]
    spread = spread_series.iloc[-1]

    mean = historical.mean()
    std = historical.std()

    if std == 0:
        return None, spread, mean, std
    z_score = (spread - mean) / std

    return z_score, spread, mean, std


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
