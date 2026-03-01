from binance.client import Client
import pandas as pd
from datetime import datetime, timezone
import time
import os
from tqdm import tqdm

client = Client()
limit_per_request = 1000


def fetch_data_from_binance(
    symbol: str, interval: str, start_str: str, end_str: str, date_format: str
) -> None:
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(root_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    filename = os.path.join(
        data_dir,
        symbol,
        f"{symbol}_{start_str.replace('.', '')}-{end_str.replace('.', '')}_{interval}.csv",
    )

    if os.path.isfile(filename):
        print(f"{filename} already exists, skipping.")
        return

    start_time = datetime.strptime(start_str, date_format).replace(tzinfo=timezone.utc)
    end_time = datetime.strptime(end_str, date_format).replace(tzinfo=timezone.utc)

    start_ts = int(start_time.timestamp() * 1000)
    end_ts = int(end_time.timestamp() * 1000)

    klines_all = []

    pbar = tqdm(
        total=100,
        desc=f"Downloading data for {symbol} ({interval}) {start_time.strftime('%Y-%m-%d')} → {end_time.strftime('%Y-%m-%d')}: ",
        bar_format="{desc} {bar} {n_fmt}% | {remaining}",
    )

    current_ts = start_ts
    while current_ts < end_ts:
        klines = client.get_klines(
            symbol=symbol,
            interval=interval,
            startTime=current_ts,
            endTime=end_ts,
            limit=limit_per_request,
        )

        if not klines:
            print(f"Missing record while fetching data for {symbol}. Fetching stopped.")
            break

        klines_all.extend(klines)
        last_close = klines[-1][6]
        current_ts = last_close + 1

        progress = min((current_ts - start_ts) / (end_ts - start_ts) * 100, 100)
        pbar.n = int(progress)
        pbar.refresh()

        time.sleep(0.1)

    pbar.close()
    print(f"Downloaded {len(klines_all)} records.")

    if not klines_all:
        print("No data downloaded. File will not be saved.")
        return

    columns = [
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
    ]

    df = pd.DataFrame(klines_all, columns=columns)

    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")
    num_cols = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_asset_volume",
        "taker_buy_base",
        "taker_buy_quote",
    ]
    df[num_cols] = df[num_cols].astype(float)

    os.makedirs(os.path.join(data_dir, symbol), exist_ok=True)
    df.to_csv(filename, index=False, encoding="utf-8-sig")
    print(f"Data saved to: {filename}")


if __name__ == "__main__":
    tickers = [
        "BTCUSDT",
        "ETHUSDT",
        "BNBUSDT",
        "XRPUSDT",
        "ADAUSDT",
        "DOGEUSDT",
        "SOLUSDT",
        "TRXUSDT",
        "DOTUSDT",
        "LTCUSDT",
        "SHIBUSDT",
        "BCHUSDT",
        "AVAXUSDT",
        "XLMUSDT",
        "LINKUSDT",
        "UNIUSDT",
        "ATOMUSDT",
        "ICPUSDT",
        "FILUSDT",
        "LDOUSDT",
        "APTUSDT",
        "QNTUSDT",
        "ARBUSDT",
        "VETUSDT",
        "MKRUSDT",
        "OPUSDT",
        "NEARUSDT",
        "GRTUSDT",
        "AAVEUSDT",
        "ALGOUSDT",
    ]

    for ticker in tickers:
        fetch_data_from_binance(
            symbol=ticker,
            interval="1h",
            start_str="01.01.2023",
            end_str="01.01.2025",
            date_format="%d.%m.%Y",
        )
