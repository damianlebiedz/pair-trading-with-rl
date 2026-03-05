import json
import os
import time
from pathlib import Path
import pandas as pd
from binance.client import Client
from omegaconf import OmegaConf
from tqdm import tqdm

from modules.utils.logger import get_logger

logger = get_logger(__name__)


def fetch_historical_data():
    """
    Downloads and compiles historical market data for the entire backtesting universe.

    This script parses the monthly asset baskets and extracts a deduplicated set of unique
    tickers. If an asset appears in multiple monthly baskets, its data is fetched only once
    for the entire global test period. This approach eliminates redundant API calls,
    minimizes network latency, and optimizes local storage.

    Key Features:
    - Precise Boundary Alignment: Enforces strict start and end timestamps (avoiding
      the "next year's first candle" spillover) to guarantee perfectly aligned DataFrames.
    - Optimized Storage: Converts raw Binance JSON responses into highly compressed,
      columnar Parquet files for lightning-fast I/O during strategy execution.

    Outputs:
        - A collection of .parquet files in the local data directory, ready for the DataLoader.
    """
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent

    config_path = project_root / "config" / "helpers" / "fetch_historical_data.yaml"
    cfg = OmegaConf.load(config_path)

    client = Client(requests_params={"timeout": cfg.timeout})

    data_dir = project_root / "data" / "historical"
    data_dir.mkdir(exist_ok=True)

    json_path = project_root / "config" / "schemas" / "list_of_assets.json"

    if not json_path.exists():
        logger.error(
            f"Error: {json_path} not found. Run generate_assets_list.py first."
        )
        return

    with open(json_path, "r", encoding="utf-8") as f:
        universes = json.load(f)

    unique_tickers = set()
    for month, tickers in universes.items():
        unique_tickers.update(tickers)

    unique_tickers = sorted(list(unique_tickers))
    total_tickers = len(unique_tickers)
    logger.info(f"Found {total_tickers} unique tickers across all months.")

    start_time = pd.Timestamp(cfg.start, tz="UTC")
    end_time = pd.Timestamp(cfg.end, tz="UTC")

    expected_rows = len(
        pd.date_range(
            start=start_time, end=end_time, freq=cfg.interval, inclusive="left"
        )
    )
    logger.debug(
        f"Target range: {start_time} to {end_time}. Expected rows per asset: {expected_rows}"
    )

    file_start_str = start_time.strftime("%Y%m%d")
    file_end_str = end_time.strftime("%Y%m%d")

    start_ts = int(start_time.timestamp() * 1000)
    end_ts = int(end_time.timestamp() * 1000) - 1

    for i, symbol in enumerate(unique_tickers, 1):
        filename = (
            data_dir
            / f"{symbol}_{cfg.interval}_{file_start_str}-{file_end_str}.parquet"
        )

        if filename.exists():
            try:
                existing_df = pd.read_parquet(filename)
                if len(existing_df) == expected_rows:
                    logger.info(
                        f"[{i}/{total_tickers}] {filename.name} is valid ({len(existing_df)} rows), skipping."
                    )
                    continue
                else:
                    logger.info(
                        f"[{i}/{total_tickers}] {filename.name} has WRONG row count ({len(existing_df)}). Redownloading..."
                    )
            except Exception as e:
                logger.info(
                    f"[{i}/{total_tickers}] {filename.name} is corrupted: {e}. Redownloading..."
                )

        klines_all = []
        current_ts = start_ts

        pbar = tqdm(
            total=100,
            desc=f"[{i}/{total_tickers}] Downloading {symbol} ({cfg.interval})",
            bar_format="{desc}: {bar} {n_fmt}% | {remaining}",
        )

        while current_ts < end_ts:
            klines = client.get_klines(
                symbol=symbol,
                interval=cfg.interval,
                startTime=current_ts,
                endTime=end_ts,
                limit=cfg.limit_per_request,
            )

            if not klines:
                break

            klines_all.extend(klines)
            last_close = klines[-1][6]
            current_ts = last_close + 1

            progress = min((current_ts - start_ts) / (end_ts - start_ts) * 100, 100)
            pbar.n = int(progress)
            pbar.refresh()

            time.sleep(0.1)

        pbar.close()

        if len(klines_all) != expected_rows:
            raise ValueError(
                f"DATA INTEGRITY ERROR for {symbol}: "
                f"Downloaded {len(klines_all)} rows, but expected EXACTLY {expected_rows}. "
                f"Check if the asset was listed on Binance during the entire period {cfg.start} - {cfg.end}."
            )

        df = pd.DataFrame(
            klines_all,
            columns=[
                "open_time",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "close_time",
                "quote_asset_volume",
                "number_of_trades",
                "taker_buy_base",
                "taker_buy_quote",
                "ignore",
            ],
        )

        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
        df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")

        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
        num_cols = ["open", "high", "low", "close", "volume", "quote_asset_volume"]
        df[num_cols] = df[num_cols].astype(float)

        temp_filename = filename.with_suffix(".tmp")
        df.to_parquet(temp_filename, index=False, engine="pyarrow")
        os.replace(temp_filename, filename)
        logger.info(f"--> Verified and saved {symbol} ({len(df)} rows)")

    downloaded_files = list(
        data_dir.glob(f"*_{cfg.interval}_{file_start_str}-{file_end_str}.parquet")
    )
    if len(downloaded_files) != total_tickers:
        missing = total_tickers - len(downloaded_files)
        raise FileNotFoundError(
            f"CRITICAL: Final validation failed! Found {len(downloaded_files)} files, expected {total_tickers}. {missing} files are missing."
        )

    logger.info(
        f"\nSUCCESS: All {total_tickers} assets downloaded and verified (exactly {expected_rows} rows each)."
    )


if __name__ == "__main__":
    fetch_historical_data()
