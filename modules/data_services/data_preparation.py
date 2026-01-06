import numpy as np
import pandas as pd


def add_c_norm_returns(df: pd.DataFrame, ticker_x: str, ticker_y: str) -> None:
    df[f"{ticker_x}_c_norm_returns"] = df[ticker_x] / df[ticker_x].iloc[0]
    df[f"{ticker_y}_c_norm_returns"] = df[ticker_y] / df[ticker_y].iloc[0]


def add_c_returns(df: pd.DataFrame, ticker_x: str, ticker_y: str) -> None:
    df[f"{ticker_x}_c_returns"] = df[ticker_x] / df[ticker_x].iloc[0] - 1
    df[f"{ticker_y}_c_returns"] = df[ticker_y] / df[ticker_y].iloc[0] - 1


def add_c_log_returns(df: pd.DataFrame, ticker_x: str, ticker_y: str) -> None:
    df[f"{ticker_x}_c_log_returns"] = np.log(df[ticker_x] / df[ticker_x].iloc[0])
    df[f"{ticker_y}_c_log_returns"] = np.log(df[ticker_y] / df[ticker_y].iloc[0])


def add_log_prices(df: pd.DataFrame, ticker_x: str, ticker_y: str) -> None:
    """Add log prices to DataFrame."""
    df[f"{ticker_x}_log"] = np.log(df[ticker_x])
    df[f"{ticker_y}_log"] = np.log(df[ticker_y])
