import json
from pathlib import Path
import pandas as pd
from binance.client import Client
from tqdm import tqdm
from omegaconf import OmegaConf

from modules.utils.logger import get_logger

logger = get_logger(__name__)


def generate_assets_list():
    """
    Generates a monthly tradable universe of Top N assets based on historical Binance volume.

    This script implements two critical filters to ensure institutional-grade data quality for backtesting:
    1. Global Data Availability Filter: Strictly excludes assets that lack complete historical
       data across the entire test period, ensuring perfectly aligned, gap-free DataFrames.
    2. Average Volume Filter: Ranks assets using an 1-month average quote volume rather
       than short-term metrics. This effectively eliminates temporary pump-and-dumps and ensures
       the selection of consistently liquid, stable, and high-cap projects.

    Outputs:
        - A JSON schema mapping each month to its respective asset basket.
        - A Markdown document detailing the selected assets and their average volumes.
    """
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent

    config_path = project_root / "config" / "helpers" / "generate_assets_list.yaml"
    cfg = OmegaConf.load(config_path)

    client = Client()

    json_out = project_root / "config" / "schemas" / "list_of_assets.json"
    md_out = project_root / "docs" / "list_of_assets.md"

    logger.info("Fetching exchange info from Binance...")
    exchange_info = client.get_exchange_info()
    symbols = []

    for s in exchange_info["symbols"]:
        if s["quoteAsset"] == "USDT" and s["status"] == "TRADING":
            base_asset = s["baseAsset"]
            if not any(blacklisted in base_asset for blacklisted in cfg.blacklist):
                is_leveraged = base_asset.endswith("UP") or base_asset.endswith("DOWN")
                if is_leveraged and base_asset not in cfg.whitelist:
                    continue
                symbols.append(s["symbol"])

    logger.debug(
        f"Found {len(symbols)} valid USDT pairs. Fetching 1d historical volume..."
    )

    start_ts_date = pd.Timestamp(cfg.start, tz="UTC")
    end_ts_date = pd.Timestamp(cfg.end, tz="UTC")

    data_start_time = (
        start_ts_date - pd.DateOffset(months=1) - pd.Timedelta(days=cfg.buffer_days)
    )
    data_end_time = end_ts_date + pd.Timedelta(days=cfg.buffer_days)

    start_ts = int(data_start_time.timestamp() * 1000)
    end_ts = int(data_end_time.timestamp() * 1000)

    volume_data = {}

    for symbol in tqdm(symbols, desc="Fetching Volume"):
        klines = client.get_klines(
            symbol=symbol,
            interval="1d",
            startTime=start_ts,
            endTime=end_ts,
            limit=cfg.limit_per_request,
        )
        if not klines:
            continue

        df = pd.DataFrame(
            klines,
            columns=[
                "open_time",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "close_time",
                "quote_volume",
                "trades",
                "taker_base",
                "taker_quote",
                "ignore",
            ],
        )
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        df["quote_volume"] = df["quote_volume"].astype(float)
        df.set_index("open_time", inplace=True)
        volume_data[symbol] = df["quote_volume"]

    expected_days = (end_ts_date - start_ts_date).days + 1

    valid_symbols = []
    for sym, vol_series in volume_data.items():
        if vol_series.empty:
            continue

        starts_ok = vol_series.index[0] <= start_ts_date
        ends_ok = vol_series.index[-1] >= end_ts_date

        period_data = vol_series.loc[start_ts_date:end_ts_date]
        density_ok = len(period_data) >= expected_days

        if starts_ok and ends_ok and density_ok:
            valid_symbols.append(sym)
        else:
            reason = "Gap in data" if not density_ok else "Range mismatch"
            logger.debug(
                f"Skipping {sym}: {reason} ({len(period_data)}/{expected_days} days)"
            )

    universes = {}
    md_universes = {}

    target_months = pd.date_range(start=start_ts_date, end=end_ts_date, freq="MS")

    for target_month in target_months:
        window_start = target_month - pd.DateOffset(months=1)
        window_end = target_month - pd.DateOffset(days=1)

        month_key = target_month.strftime("%Y-%m")
        monthly_volumes = {}

        for sym in valid_symbols:
            vol_series = volume_data[sym]
            period_vol = vol_series.loc[window_start:window_end]

            if len(period_vol) >= 28:
                monthly_volumes[sym] = period_vol.mean()

        sorted_symbols = sorted(
            monthly_volumes.keys(), key=lambda x: monthly_volumes[x], reverse=True
        )

        top_symbols_with_vol = [
            (sym, monthly_volumes[sym]) for sym in sorted_symbols[: cfg.top_n]
        ]
        top_symbols = [sym for sym, vol in top_symbols_with_vol]

        universes[month_key] = top_symbols
        md_universes[month_key] = top_symbols_with_vol

    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(universes, f, indent=4)
    logger.info(f"Saved JSON configuration: {json_out}")

    md_content = f"# Traded Assets Universe (Top {cfg.top_n} by Volume)\n\n"
    md_content += "This document lists the tradable universe for each month. The selection is based on the **1-month Average** daily quote volume to favor stable, high-cap projects.\n\n"

    for month_key, symbols_with_vol in md_universes.items():
        md_content += f"### {month_key}\n"
        formatted_symbols = []
        for sym, vol in symbols_with_vol:
            vol_in_millions = vol / 1_000_000
            formatted_symbols.append(f"{sym} (${vol_in_millions:.1f}M)")
        md_content += f"**Assets:** {', '.join(formatted_symbols)}\n\n"

    with open(md_out, "w", encoding="utf-8") as f:
        f.write(md_content)
    logger.info(f"Saved Markdown documentation: {md_out}")


if __name__ == "__main__":
    generate_assets_list()
