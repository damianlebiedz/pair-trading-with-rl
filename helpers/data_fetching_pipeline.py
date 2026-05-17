"""
Data Fetching & Processing Pipeline

This script serves as a comprehensive data pipeline for downloading, verifying,
and processing historical market data (klines) from Binance Data Vision (AWS S3)
specifically for the USD-M Futures market.

Key Features & Design Principles:
1. Zero Survivorship Bias: Retrieves the complete list of assets from the S3 archive,
   including delisted or bankrupt projects.
2. Strict Data Integrity: Rejects assets with missing data points (e.g., hourly gaps)
   in the Pair Selection period, automatically replacing them with the next most liquid market.
3. Fault Tolerance & Local Caching: Utilizes a local disk cache (.cache) storing data
   in Parquet format. Each month/asset is downloaded exactly once. In case of interruption,
   the script resumes instantly without redownloading.
4. Out-of-Sample Ready: Automatically aligns data to train and test windows, outputting
   continuous Parquet files ready to be ingested.
"""

import json
import io
import shutil
import zipfile
import requests
import time
import pandas as pd
import xml.etree.ElementTree as ET
from pathlib import Path
from tqdm import tqdm
from dateutil.relativedelta import relativedelta
from omegaconf import OmegaConf, DictConfig

from modules.utils.logger import get_logger
from runners.core.utils import generate_date_lists

logger = get_logger(__name__)

BASE_VISION_URL = "https://data.binance.vision/data/futures/um/monthly/klines"
S3_BUCKET_URL = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision?delimiter=/&prefix=data/futures/um/monthly/klines/"

CSV_COLUMNS = [
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


def get_all_futures_symbols() -> list[str]:
    """
    Retrieves the complete list of all historical and active USDT Futures pairs
    directly from the XML tree of the Binance Data Vision AWS S3 bucket.

    Returns:
        list: A list of symbol strings (e.g., ['BTCUSDT', 'ETHUSDT', ...]).
    """
    logger.info(
        "Downloading list of all Futures assets from Binance Data Vision archive..."
    )
    for attempt in range(3):
        try:
            response = requests.get(S3_BUCKET_URL, timeout=15)
            response.raise_for_status()
            r = ET.fromstring(response.content)
            ns = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}

            symbols = [
                prefix.text.split("/")[-2]
                for prefix in r.findall("s3:CommonPrefixes/s3:Prefix", ns)
                if prefix.text.split("/")[-2].endswith("USDT")
            ]
            logger.info(f"Found {len(symbols)} assets to USDT.")
            return symbols
        except Exception as e:
            logger.warning(
                f"Error during list downloading (Retry {attempt + 1}/3): {e}"
            )
            time.sleep(3)
    raise ConnectionError("Failed to connect with Binance Data Vision.")


def download_and_cache_month(
    symbol: str,
    interval: str,
    year: int,
    month: int,
    cache_dir: Path,
    max_retries: int = 5,
) -> pd.DataFrame | None:
    """
    Downloads a ZIP file containing klines for a specific asset and month,
    cleans the data, and saves it as an optimized .parquet file in the local cache.

    If the file already exists in the cache, it loads it directly from the disk,
    bypassing the network request to allow instant pipeline resumption.

    Args:
        symbol (str): Trading pair symbol (e.g., 'BTCUSDT').
        interval (str): Kline interval (e.g., '1h', '1d').
        year (int): Year of the data.
        month (int): Month of the data.
        cache_dir (Path): Directory to store the cached files.
        max_retries (int): Maximum number of download attempts.

    Returns:
        pd.DataFrame or None: DataFrame containing the monthly data, or None if the file is not found.
    """
    month_str = f"{month:02d}"
    cache_file = cache_dir / f"{symbol}_{interval}_{year}_{month_str}.parquet"

    if cache_file.exists():
        return pd.read_parquet(cache_file)

    file_name = f"{symbol}-{interval}-{year}-{month_str}.zip"
    url = f"{BASE_VISION_URL}/{symbol}/{interval}/{file_name}"

    for attempt in range(max_retries):
        try:
            resp = requests.get(url, timeout=30)
            if resp.status_code == 404:
                return None

            resp.raise_for_status()

            with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
                csv_name = z.namelist()[0]
                with z.open(csv_name) as f:
                    df = pd.read_csv(
                        f, header=None, names=CSV_COLUMNS, low_memory=False
                    )

                    df = df[df["open_time"] != "open_time"].copy()
                    df["open_time"] = pd.to_numeric(df["open_time"])
                    df["close_time"] = pd.to_numeric(df["close_time"])

                    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
                    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")

                    df.to_parquet(cache_file, index=False, engine="pyarrow")
                    return df

        except requests.exceptions.RequestException as e:
            logger.debug(
                f"Network error during {file_name} (Retry {attempt + 1}/{max_retries}): {e}"
            )
            time.sleep(2**attempt)
        except zipfile.BadZipFile:
            logger.debug(f"Corrupted ZIP file on the server: {file_name}")
            return None
        except Exception as e:
            logger.debug(f"Unexpected error during {file_name} unpacking: {e}")
            return None

    return None


def fetch_data_for_period(
    symbol: str,
    interval: str,
    start_dt: pd.Timestamp,
    end_dt: pd.Timestamp,
    cache_dir: Path,
) -> pd.DataFrame:
    """
    Stitches together the required monthly cached files into a single continuous
    DataFrame and precisely crops it to the specified datetime window.

    Args:
        symbol (str): Trading pair symbol.
        interval (str): Kline interval.
        start_dt (datetime): Start datetime of the required period.
        end_dt (datetime): End datetime of the required period.
        cache_dir (Path): Directory where cached files are stored.

    Returns:
        pd.DataFrame: Continuous price data for the specified period.
    """
    dfs = []
    current_date = start_dt.replace(day=1, hour=0, minute=0, second=0)

    while current_date <= end_dt:
        df_month = download_and_cache_month(
            symbol, interval, current_date.year, current_date.month, cache_dir
        )
        if df_month is not None and not df_month.empty:
            dfs.append(df_month)
        current_date += relativedelta(months=1)

    if not dfs:
        return pd.DataFrame()

    full_df = pd.concat(dfs, ignore_index=True)
    mask = (full_df["open_time"] >= start_dt) & (full_df["open_time"] < end_dt)
    return full_df.loc[mask].copy()


def generate_and_fetch_pipeline(
    project_root: Path, cfg: DictConfig, interval: str
) -> None:
    """
    The main orchestrator of the data fetching and processing pipeline.

    Workflow:
    1. Iterates through the defined backtest time windows (iterations).
    2. Ranks all available assets based on their average daily volume.
    3. Verifies strict data integrity (no missing hours) for the Top N assets.
    4. Saves the universe configurations to JSON and Markdown files.
    5. Merges the verified cached months into final, continuous Parquet files for the RL model.

    Args:
        project_root (Path): The root directory of the project.
        cfg (DictConfig): Configuration dictionary loaded from YAML.
        interval (str): The main kline interval to fetch (e.g., '1h').
    """
    data_dir = project_root / "data" / "historical"
    cache_dir = project_root / "data" / "historical" / ".cache"

    data_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    json_out = project_root / "config" / "schemas" / "list_of_assets.json"
    md_out = project_root / "docs" / "list_of_assets.md"

    top_n = cfg.top_n

    config_dict = {
        "start": cfg.start,
        "end": cfg.end,
        "test_end": cfg.test_end,
    }
    lists = generate_date_lists(config_dict, cfg.iterations)
    all_symbols = get_all_futures_symbols()

    universes = {}

    if json_out.exists():
        with open(json_out, "r", encoding="utf-8") as f:
            universes = json.load(f)

    for i in range(cfg.iterations):
        vol_start = pd.Timestamp(lists["start_list"][i])
        vol_end = pd.Timestamp(lists["end_list"][i])
        data_end = pd.Timestamp(lists["test_end_list"][i])
        month_key = vol_start.strftime("%Y-%m")

        if month_key in universes:
            logger.info(
                f"Iter {i + 1}/{cfg.iterations} ({month_key}) already done. Skipping."
            )
            continue

        logger.info(
            f"\nIter {i + 1}/{cfg.iterations}: Month key {month_key} | Volume {vol_start.date()} - {vol_end.date()} | {interval} data end: {data_end.date()}"
        )

        monthly_volumes = {}
        expected_days = (vol_end - vol_start).days

        for sym in tqdm(
            all_symbols, desc=f"Iter {i + 1}/{cfg.iterations} | Volume Fetching"
        ):
            if any(blacklisted in sym for blacklisted in cfg.blacklist):
                continue

            df_1d = fetch_data_for_period(sym, "1d", vol_start, vol_end, cache_dir)
            if not df_1d.empty and len(df_1d) >= expected_days * 0.95:
                avg_vol = df_1d["quote_asset_volume"].astype(float).mean()
                monthly_volumes[sym] = avg_vol

        sorted_symbols = sorted(
            monthly_volumes.keys(), key=lambda x: monthly_volumes[x], reverse=True
        )

        expected_formation_index = pd.date_range(
            start=vol_start, end=vol_end, freq=interval, inclusive="left"
        )
        expected_formation_hours = len(expected_formation_index)

        expected_full_index = pd.date_range(
            start=vol_start, end=data_end, freq=interval, inclusive="left"
        )
        expected_full_hours = len(expected_full_index)

        logger.info("Validating data completeness (must survive formation window)...")

        accepted_assets = []
        broken_assets = []

        for sym in sorted_symbols:
            if len(accepted_assets) >= top_n:
                break

            df_interval = fetch_data_for_period(
                sym, interval, vol_start, data_end, cache_dir
            )
            if df_interval.empty:
                continue

            df_interval = df_interval.drop_duplicates(subset=["open_time"])

            df_formation = df_interval[df_interval["open_time"] < vol_end]

            if len(df_formation) == expected_formation_hours:
                accepted_assets.append(sym)
                if len(df_interval) < expected_full_hours:
                    broken_assets.append(sym)
                    logger.debug(
                        f"  [+] Accepted (but delisted in test): {sym} ({len(accepted_assets)}/{top_n})"
                    )
                else:
                    logger.debug(
                        f"  [+] Accepted: {sym} ({len(accepted_assets)}/{top_n})"
                    )
            else:
                missing = expected_formation_hours - len(df_formation)
                logger.debug(
                    f"  [-] Rejected: {sym} (missing {missing}h in formation window)"
                )

        if len(accepted_assets) < top_n:
            logger.warning(
                f"[Lack of data] Found complete data for ONLY {len(accepted_assets)}/{top_n} assets ({month_key}) in iteration {i + 1}/{cfg.iterations}!"
            )
        else:
            logger.info(
                f"Found complete data for {len(accepted_assets)}/{top_n} assets. {len(broken_assets)} delisted during the test window."
            )

        universes[month_key] = {
            "volume_window_start": vol_start.strftime("%Y-%m-%d %H:%M:%S"),
            "volume_window_end": vol_end.strftime("%Y-%m-%d %H:%M:%S"),
            "data_fetch_start": vol_start.strftime("%Y-%m-%d %H:%M:%S"),
            "data_fetch_end": data_end.strftime("%Y-%m-%d %H:%M:%S"),
            "requested_count": top_n,
            "found_count": len(accepted_assets),
            "broken_in_test": broken_assets,
            "is_complete": len(accepted_assets) == top_n,
            "assets": accepted_assets,
        }
        with open(json_out, "w", encoding="utf-8") as f:
            json.dump(universes, f, indent=4)

    with open(json_out, "r", encoding="utf-8") as f:
        final_universes = json.load(f)

    symbol_ranges = {}
    for m_key, u_data in final_universes.items():
        s_dt = pd.Timestamp(u_data["data_fetch_start"])
        e_dt = pd.Timestamp(u_data["data_fetch_end"])
        for sym in u_data["assets"]:
            if sym not in symbol_ranges:
                symbol_ranges[sym] = []
            symbol_ranges[sym].append((s_dt, e_dt))

    for sym, ranges in symbol_ranges.items():
        sym_dfs = []
        global_min_dt = min([r[0] for r in ranges])
        global_max_dt = max([r[1] for r in ranges])

        for s_dt, e_dt in ranges:
            df_part = fetch_data_for_period(sym, interval, s_dt, e_dt, cache_dir)
            sym_dfs.append(df_part)

        final_df = pd.concat(sym_dfs, ignore_index=True)
        final_df = final_df.drop_duplicates(subset=["open_time"]).sort_values(
            "open_time"
        )

        full_index = pd.date_range(
            start=global_min_dt, end=global_max_dt, freq=interval, inclusive="left"
        )
        final_df = (
            final_df.set_index("open_time")
            .reindex(full_index)
            .rename_axis("open_time")
            .reset_index()
        )

        num_cols = ["open", "high", "low", "close", "volume", "quote_asset_volume"]
        final_df[num_cols] = final_df[num_cols].astype(float)

        file_start_str = global_min_dt.strftime("%Y%m%d")
        file_end_str = global_max_dt.strftime("%Y%m%d")

        for old_file in list(data_dir.glob(f"{sym}_{interval}_*.parquet")):
            old_file.unlink()

        filename = (
            data_dir / f"{sym}_{interval}_{file_start_str}-{file_end_str}.parquet"
        )
        final_df.to_parquet(filename, index=False, engine="pyarrow")
        logger.debug(f"Saved file: {filename.name}")

    logger.info("Parquets downloaded successfully.")

    try:
        shutil.rmtree(cache_dir)
        logger.info("Deleted .cache folder.")
    except Exception as e:
        logger.warning(f"Failed to delete .cache folder: {e}")

    logger.info(f"Generating Markdown documentation: {md_out}")
    md_content = f"# Traded Assets Universe (Top {cfg.top_n})\n\n"
    md_content += (
        "**Methodology Note:** Universes generated from **Binance Data Vision (Futures USD-M)**. "
        "Assets selected based on highest average daily quote volume. Assets with any data gaps in the "
        f"{interval} timeframe IN THE PAIR SELECTION PERIOD were automatically excluded and replaced by the next highest volume asset to ensure data integrity.\n\n"
        "--- \n\n"
    )

    for i, (m_key, u_data) in enumerate(final_universes.items()):
        v_start = u_data["volume_window_start"][:10]
        v_end = u_data["volume_window_end"][:10]
        d_end = u_data["data_fetch_end"][:10]
        assets = u_data["assets"]

        found = u_data.get("found_count", len(assets))
        req = u_data.get("requested_count", cfg.top_n)
        broken = u_data.get("broken_in_test", [])

        md_content += f"### Iteration {i + 1} (Key: {m_key})\n"
        md_content += f"- **Volume Period:** {v_start} to {v_end}\n"
        md_content += f"- **Full Data Window:** {v_start} to {d_end}\n\n"

        if found < req:
            md_content += f"> **DATA WARNING:** Found complete historical data for only **{found}/{req}** assets in the formation window. Universe is smaller than requested.\n\n"

        if broken:
            md_content += f"> **SURVIVORSHIP NOTE:** Out of {found} accepted assets, **{len(broken)}** stopped trading during the test phase ({', '.join(broken)}). Their missing rows are padded with `NaN`s to ensure predictable DataFrame shapes.\n\n"

        md_content += f"**Final Assets:** {', '.join(assets)}\n\n"

    with open(md_out, "w", encoding="utf-8") as f:
        f.write(md_content)

    logger.info("Pipeline completed successfully.")


if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent
    root = script_dir.parent
    cfg_list = OmegaConf.load(
        root / "config" / "helpers" / "data_fetching_pipeline.yaml"
    )
    cfg_base_list = OmegaConf.load(root / "config" / "base.yaml")
    base_interval = cfg_base_list["market"]["interval"]

    generate_and_fetch_pipeline(project_root=root, cfg=cfg_list, interval=base_interval)
