import json
from pathlib import Path
import pandas as pd
from binance.client import Client
from tqdm import tqdm
from omegaconf import OmegaConf

from modules.utils.logger import get_logger
from runners.core.utils import generate_date_lists

logger = get_logger(__name__)


def generate_assets_list():
    """Generates dynamic asset universes."""
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent

    config_path = project_root / "config" / "helpers" / "generate_assets_list.yaml"
    cfg = OmegaConf.load(config_path)

    json_out = project_root / "config" / "schemas" / "list_of_assets.json"
    md_out = project_root / "docs" / "list_of_assets.md"

    config_dict = {
        "start": cfg.start,
        "end": cfg.end,
        "test_end": cfg.test_end,
    }
    iterations = cfg.iterations
    top_n = cfg.top_n

    lists = generate_date_lists(config_dict, iterations)
    ps_start_list = [pd.Timestamp(d) for d in lists["start_list"]]
    ps_end_list = [pd.Timestamp(d) for d in lists["end_list"]]
    test_end_list = [pd.Timestamp(d) for d in lists["test_end_list"]]

    global_start = min(ps_start_list) - pd.Timedelta(days=5)
    global_end = max(ps_end_list) + pd.Timedelta(days=1)

    client = Client()

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

    start_ts = int(global_start.timestamp() * 1000)
    end_ts = int(global_end.timestamp() * 1000)

    volume_data = {}
    for symbol in tqdm(symbols, desc="Fetching Volume"):
        klines = client.get_historical_klines(
            symbol=symbol, interval="1d", start_str=start_ts, end_str=end_ts
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
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
        df["quote_volume"] = df["quote_volume"].astype(float)
        df.set_index("open_time", inplace=True)
        volume_data[symbol] = df["quote_volume"]

    valid_symbols = [sym for sym, vol in volume_data.items() if not vol.empty]

    universes = {}
    md_universes = {}

    for i in range(iterations):
        window_start = ps_start_list[i]
        window_end = ps_end_list[i]
        iter_test_end = test_end_list[i]

        month_key = window_start.strftime("%Y-%m")
        range_label = f"Iteration {i + 1}: {window_start.strftime('%d %b %Y')} - {window_end.strftime('%d %b %Y')}"

        monthly_volumes = {}

        expected_days = (window_end - window_start).days

        for sym in valid_symbols:
            vol_series = volume_data[sym]

            mask = (vol_series.index >= window_start) & (vol_series.index < window_end)
            period_vol = vol_series.loc[mask]

            day_before = window_start - pd.Timedelta(days=1)
            has_prior_day = day_before in vol_series.index

            if len(period_vol) == expected_days and has_prior_day:
                monthly_volumes[sym] = period_vol.mean()

        sorted_symbols = sorted(
            monthly_volumes.keys(), key=lambda x: monthly_volumes[x], reverse=True
        )
        top_with_vol = [(sym, monthly_volumes[sym]) for sym in sorted_symbols[:top_n]]

        universes[month_key] = {
            "volume_window_start": window_start.strftime("%Y-%m-%d %H:%M:%S"),
            "volume_window_end": window_end.strftime("%Y-%m-%d %H:%M:%S"),
            "data_fetch_start": window_start.strftime("%Y-%m-%d %H:%M:%S"),
            "data_fetch_end": iter_test_end.strftime("%Y-%m-%d %H:%M:%S"),
            "assets": [sym for sym, vol in top_with_vol],
        }

        md_universes[range_label] = (
            month_key,
            window_start,
            window_end,
            iter_test_end,
            top_with_vol,
        )

    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(universes, f, indent=4)
    logger.info(f"Saved JSON configuration: {json_out}")

    md_content = f"# Traded Assets Universe (Top {top_n})\n\n"
    md_content += (
        "**Methodology Note:** The universes below were automatically generated based on the settings "
        "in `generate_assets_list.yaml`. For each iteration, the average daily quote volume (USDT) is calculated "
        "strictly within the defined `Volume Window` (which corresponds to the pair_selection period). "
        "The `Data Fetch Window` indicates the full period required by the backtester (including the out-of-sample test).\n\n"
    )
    md_content += "---\n\n"

    for range_label, (
        m_key,
        v_start,
        v_end,
        df_end,
        symbols_with_vol,
    ) in md_universes.items():
        md_content += f"### {range_label} (Key: {m_key})\n"
        md_content += f"- **Volume Calculation:** {v_start.strftime('%Y-%m-%d')} to {v_end.strftime('%Y-%m-%d')}\n"
        md_content += f"- **Data Required Until:** {df_end.strftime('%Y-%m-%d')}\n\n"

        formatted = [f"{sym} (${vol / 1e6:.1f}M)" for sym, vol in symbols_with_vol]
        md_content += f"**Assets:** {', '.join(formatted)}\n\n"

    with open(md_out, "w", encoding="utf-8") as f:
        f.write(md_content)
    logger.info(f"Saved Markdown documentation: {md_out}")


if __name__ == "__main__":
    generate_assets_list()
