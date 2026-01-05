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
    mean = spread_series.mean()
    std = spread_series.std()
    spread = spread_series.iloc[-1]
    if std == 0:
        return None
    z_score = (spread - mean) / std

    return z_score


def calculate_half_life_window(
        x_col: str,
        y_col: str,
        beta: float,
        df: pd.DataFrame,
        window_factor: float = 1.0,
) -> int | None:
    """
    Oblicza Half-Life DOKŁADNIE TEGO spreadu, którym handlujesz.
    Wymaga podania bety używanej w strategii.
    """
    # Budujemy spread tak samo jak w strategii
    series = df[x_col] - (beta * df[y_col])

    # Zabezpieczenie: minimum danych
    if len(series) < 10:
        raise Exception("len(series) < 10")

    lag = series.shift(1)
    ret = series - lag

    lag = lag.iloc[1:]
    ret = ret.iloc[1:]

    try:
        X = sm.add_constant(lag)
        model = sm.OLS(ret, X, missing="drop").fit()
        lam = model.params.iloc[1]

        # Jeśli lambda >= 0, Twój spread trenduje (nawet jeśli inna beta byłaby lepsza,
        # to Twoja obecna beta nie daje mean-reversion) -> Nie handluj.
        if lam >= 0:
            return None

        half_life = -np.log(2) / lam
        window = int(half_life * window_factor)

        return max(5, window)

    except Exception as e:
        print(e)
        return None
