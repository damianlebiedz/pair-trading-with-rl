from pathlib import Path
import pandas as pd

from modules.core.enums import Interval


def get_project_root() -> Path:
    """
    Returns the absolute path to the project root directory.
    Assumes this file is located at: <PROJECT_ROOT>/modules/data_services/data_loaders.py
    """
    return Path(__file__).resolve().parents[2]


def load_single_ticker(
    ticker: str, start: str, end: str, interval: Interval, base_dir: Path
) -> pd.DataFrame:
    """Load data for a single asset and return as a DataFrame."""
    ticker_dir = base_dir / ticker

    if not ticker_dir.exists():
        raise FileNotFoundError(f"Directory not found: {ticker_dir}")

    files = list(ticker_dir.glob(f"*_{interval.value}.csv"))
    if not files:
        raise FileNotFoundError(
            f"No CSV file with interval '{interval.value}' found in {ticker_dir}"
        )

    df = pd.read_csv(files[0], parse_dates=["open_time", "close_time"])

    first_date = df["open_time"].min()
    last_date = df["open_time"].max()

    if first_date > pd.Timestamp(start) or last_date < pd.Timestamp(end):
        raise ValueError(
            f"Data for {ticker} not found for full {start}-{end} date range. Available: {first_date} to {last_date}"
        )

    return df.set_index("open_time")[["close"]].rename(columns={"close": ticker})


def load_data(
    tickers: list[str], start: str, end: str, interval: Interval, data_dir: str = "data"
) -> pd.DataFrame:
    """Load data for a list of assets and return as DataFrame."""
    base_dir = get_project_root() / data_dir

    dfs = [load_single_ticker(t, start, end, interval, base_dir) for t in tickers]

    data = pd.concat(dfs, axis=1)
    data = data[(data.index > start) & (data.index <= end)]

    if data.empty:
        raise ValueError(
            f"No data available for tickers {tickers} in range {start} to {end}"
        )

    return data


def load_pair(x: str, y: str, *args, **kwargs) -> pd.DataFrame:
    """Load data for a single pair and return as DataFrame."""
    return load_data([x, y], *args, **kwargs)
