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
    data_start_time = start_ts_date - pd.Timedelta(days=cfg.buffer_days)
    data_end_time = (
        end_ts_date
        + pd.DateOffset(months=cfg.window_months)
        + pd.Timedelta(days=cfg.buffer_days)
    )

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

    valid_symbols = [sym for sym, vol in volume_data.items() if not vol.empty]

    universes = {}
    md_universes = {}

    target_dates = pd.date_range(start=start_ts_date, end=end_ts_date, freq="MS")

    for target_date in target_dates:
        window_start = target_date
        window_end = (
            target_date + pd.DateOffset(months=cfg.window_months) - pd.Timedelta(days=1)
        )

        month_key = target_date.strftime("%Y-%m")
        range_label = (
            f"{window_start.strftime('%b %Y')} - {window_end.strftime('%b %Y')}"
        )
        monthly_volumes = {}

        for sym in valid_symbols:
            vol_series = volume_data[sym]
            period_vol = vol_series.loc[window_start:window_end]
            if len(period_vol) >= (28 * cfg.window_months):
                monthly_volumes[sym] = period_vol.mean()

        sorted_symbols = sorted(
            monthly_volumes.keys(), key=lambda x: monthly_volumes[x], reverse=True
        )
        top_with_vol = [
            (sym, monthly_volumes[sym]) for sym in sorted_symbols[: cfg.top_n]
        ]

        universes[month_key] = [sym for sym, vol in top_with_vol]
        md_universes[range_label] = (month_key, top_with_vol)

    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(universes, f, indent=4)
    logger.info(f"Saved JSON configuration: {json_out}")

    md_content = f"# Traded Assets Universe (Top {cfg.top_n})\n\n"
    md_content += (
        f"**Methodology Note:** This universe is selected based on the **average daily quote volume (USDT)** "
        f"calculated over a **{cfg.window_months}-month forward-looking formation window**. "
        f"This ensures that the assets used for pair selection are among the most liquid projects "
        f"specifically during the formation and testing periods. Assets with significant data gaps "
        f"(less than 27 days per month) are automatically excluded.\n\n"
    )
    md_content += "---\n\n"
    for range_label, (m_key, symbols_with_vol) in md_universes.items():
        md_content += f"### {range_label} (Key: {m_key})\n"
        formatted = [f"{sym} (${vol / 1e6:.1f}M)" for sym, vol in symbols_with_vol]
        md_content += f"**Assets:** {', '.join(formatted)}\n\n"

    with open(md_out, "w", encoding="utf-8") as f:
        f.write(md_content)
    logger.info(f"Saved Markdown documentation: {md_out}")


if __name__ == "__main__":
    generate_assets_list()
