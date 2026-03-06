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
    Downloads and compiles historical market data ONLY for the active periods of each ticker.
    Instead of downloading the entire global backtest range for every coin, it calculates
    the precise minimum start and maximum end date required for each asset based on the iterations
    it appears in within the list_of_assets.json file.
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

    ticker_ranges = {}
    for month_key, data in universes.items():
        iter_start = pd.Timestamp(data["data_fetch_start"])
        iter_end = pd.Timestamp(data["data_fetch_end"])

        for ticker in data["assets"]:
            if ticker not in ticker_ranges:
                ticker_ranges[ticker] = {"start": iter_start, "end": iter_end}
            else:
                ticker_ranges[ticker]["start"] = min(
                    ticker_ranges[ticker]["start"], iter_start
                )
                ticker_ranges[ticker]["end"] = max(
                    ticker_ranges[ticker]["end"], iter_end
                )

    total_tickers = len(ticker_ranges)
    logger.info(
        f"Found {total_tickers} unique tickers. Downloading dynamic timeframes..."
    )

    interval = cfg.interval

    for i, (symbol, date_range) in enumerate(sorted(ticker_ranges.items()), 1):
        start_time = date_range["start"]
        end_time = date_range["end"]

        expected_index = pd.date_range(
            start=start_time, end=end_time, freq=interval, inclusive="left"
        )
        expected_rows = len(expected_index)

        file_start_str = start_time.strftime("%Y%m%d")
        file_end_str = end_time.strftime("%Y%m%d")
        filename = (
            data_dir / f"{symbol}_{interval}_{file_start_str}-{file_end_str}.parquet"
        )

        old_files = list(data_dir.glob(f"{symbol}_{interval}_*.parquet"))
        for old_file in old_files:
            if old_file.name != filename.name:
                old_file.unlink()

        if filename.exists():
            try:
                existing_df = pd.read_parquet(filename)
                if len(existing_df) == expected_rows:
                    logger.info(
                        f"[{i}/{total_tickers}] {filename.name} is valid, skipping."
                    )
                    continue
            except Exception as e:
                logger.debug(e)
                pass

        start_ts = int(start_time.timestamp() * 1000)
        end_ts = int(end_time.timestamp() * 1000) - 1
        current_ts = start_ts
        klines_all = []

        pbar = tqdm(
            total=100,
            desc=f"[{i}/{total_tickers}] {symbol} ({start_time.strftime('%b%y')}-{end_time.strftime('%b%y')})",
            bar_format="{desc}: {bar} {n_fmt}% | {remaining}",
        )

        while current_ts < end_ts:
            klines = client.get_klines(
                symbol=symbol,
                interval=interval,
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

        if not df.empty:
            df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
            df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")
            num_cols = ["open", "high", "low", "close", "volume", "quote_asset_volume"]
            df[num_cols] = df[num_cols].astype(float)

            df.set_index("open_time", inplace=True)
            df = df[~df.index.duplicated(keep="last")]
            df = df.reindex(expected_index)
            df.reset_index(inplace=True)
            df.rename(columns={"index": "open_time"}, inplace=True)
        else:
            logger.warning(
                f"[{i}/{total_tickers}] No data returned for {symbol}. Padding with NaNs."
            )
            df = pd.DataFrame({"open_time": expected_index})
            for col in [
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
            ]:
                df[col] = float("nan")

        temp_filename = filename.with_suffix(".tmp")
        df.to_parquet(temp_filename, index=False, engine="pyarrow")
        os.replace(temp_filename, filename)

    logger.info("\nSUCCESS: All specific timeframe assets downloaded.")


if __name__ == "__main__":
    fetch_historical_data()
