from pathlib import Path
import pandas as pd

from modules.core.enums import Interval
from modules.utils.logger import get_logger

logger = get_logger(__name__)


def get_project_root() -> Path:
    """
    Returns the absolute path to the project root directory.
    Assumes this file is located at: <PROJECT_ROOT>/modules/data_services/data_loaders.py
    """
    return Path(__file__).resolve().parents[2]


def load_single_ticker(
    ticker: str, start: str, end: str, interval: Interval, base_dir: Path
) -> pd.DataFrame:
    """
    Load data for a single asset and return as a DataFrame.

    To strictly prevent look-ahead bias and support the "Signal on Close, Execute on Open" (SoC-EoO)
    execution model, this loader alters the default Binance timestamping logic.

    - Raw Binance data labels candles by `open_time` (e.g., 00:00:00 for the 00:00-00:59 candle).
    - This loader extracts the `close_time` (e.g., 00:59:59.999) and rounds it up to the nearest
      second (e.g., 01:00:00) to act as the DataFrame's main index.

    Result: A row indexed at `01:00:00` contains the `close` price that was finalized exactly
    at that precise moment (the culmination of the 00:00-01:00 trading hour). When the strategy
    loop processes the `01:00:00` step, it is accessing strictly historical, fully-formed data.
    """
    if not base_dir.exists():
        raise FileNotFoundError(f"Directory not found: {base_dir}")

    files = list(base_dir.glob(f"{ticker}_{interval.value}_*.parquet"))
    if not files:
        raise FileNotFoundError(
            f"No parquet file with interval '{interval.value}' found for {ticker} in {base_dir}"
        )

    df = pd.read_parquet(files[0])

    df["timestamp"] = df["close_time"].dt.ceil("s")

    start_dt = pd.Timestamp(start)
    end_dt = pd.Timestamp(end)

    expected_first = (
        start_dt + pd.Timedelta(hours=1)
        if interval.value == "1h"
        else start_dt + pd.Timedelta(days=1)
    )

    actual_first = df["timestamp"].min()
    actual_last = df["timestamp"].max()

    if actual_first > expected_first:
        raise ValueError(
            f"Data for {ticker} not found! "
            f"Required: {expected_first} to {end_dt}. "
            f"Available in file: {actual_first} to {actual_last}"
        )

    df = df.set_index("timestamp")[["close"]].rename(columns={"close": ticker})

    return df.loc[expected_first:end_dt]


def load_data(
    tickers: list[str],
    start: str,
    end: str,
    interval: Interval,
    data_dir: str = "data/historical",
) -> pd.DataFrame:
    """Load data for a list of assets and return as DataFrame."""
    base_dir = get_project_root() / data_dir

    dfs = []

    for t in tickers:
        df = load_single_ticker(t, start, end, interval, base_dir)

        df = df.loc[~df.index.duplicated(keep="last")]
        df_subset = df[(df.index > start) & (df.index <= end)].copy()

        if df_subset.empty:
            logger.warning(f"Rejected {t}: lack of data in {start} - {end}")
            continue

        if df_subset.isna().any().any():
            logger.warning(
                f"{t} contains missing data (likely delisted) in window {start} - {end}. NaNs preserved."
            )

        dfs.append(df_subset)

    try:
        data = pd.concat(dfs, axis=1)
    except Exception as e:
        logger.error("Error during pd.concat")
        raise e

    data = data[(data.index > start) & (data.index <= end)]

    if data.empty:
        raise ValueError(f"All assets were rejected ({start} to {end})")

    return data


def load_pair(x: str, y: str, *args, **kwargs) -> pd.DataFrame:
    """Load data for a single pair and return as DataFrame."""
    return load_data([x, y], *args, **kwargs)
