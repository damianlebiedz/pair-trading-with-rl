from functools import reduce
from io import StringIO
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import json

from modules.core.enums import Interval, Source
from modules.performance.models import StrategyResult
from modules.data_services.data_loaders import load_data, get_project_root


def get_steps(
    interval: Interval,
) -> int:
    """Get steps of the interval."""
    if interval == Interval.D1:
        return 1
    elif interval == Interval.H4:
        return 6
    elif interval == Interval.H1:
        return 24
    elif interval == Interval.M30:
        return 48
    elif interval == Interval.M15:
        return 96
    elif interval == Interval.M5:
        return 288
    elif interval == Interval.M3:
        return 480
    elif interval == Interval.M1:
        return 1440
    else:
        raise ValueError(f"Wrong interval '{interval}', should be in: {Interval}")


def add_log_prices(df: pd.DataFrame, ticker_x: str, ticker_y: str) -> None:
    """Add log prices to DataFrame."""
    df[f"{ticker_x}_{Source.LOG.value}"] = np.log(df[ticker_x])
    df[f"{ticker_y}_{Source.LOG.value}"] = np.log(df[ticker_y])


def merge_by_pair(dfs: list[pd.DataFrame], keep_cols: list[list[str]]) -> pd.DataFrame:
    """Merge dataframes from statistical tests into one dataframe."""
    trimmed = []
    for df, cols in zip(dfs, keep_cols):
        trimmed.append(df[["pair"] + cols])

    merged = reduce(
        lambda left, right: pd.merge(left, right, on="pair", how="outer"), trimmed
    )
    return merged


def load_btc_benchmark(
    test_start: str,
    test_end: str,
    interval: Interval,
    fee_rate: float,
) -> pd.DataFrame:
    """
    Generates a Bitcoin buy-and-hold benchmark.

    The benchmark represents a passive long-only investment in Bitcoin (BTC/USDT)
    over the test period. Returns are calculated from the BTC price series by
    computing period-to-period percentage changes and compounding them to obtain
    the cumulative return (equity curve). Transaction costs (fee_rate) are applied
    at the entry (initial purchase) and exit (final liquidation) of the investment.

    This benchmark serves as a simple market reference for comparing the strategy's
    performance against the dominant asset in the cryptocurrency market.
    """
    btc_data = load_data(
        tickers=["BTCUSDT"],
        start=test_start,
        end=test_end,
        interval=interval,
    )

    invested_data = (btc_data["BTCUSDT"] / btc_data["BTCUSDT"].iloc[0]) * (1 - fee_rate)
    invested_data.iloc[-1] *= 1 - fee_rate

    btc_data["BTC_return"] = invested_data - 1.0
    btc_data["BTC_pct"] = invested_data.pct_change()
    btc_data.loc[btc_data.index[0], "BTC_pct"] = 0.0

    return btc_data


def load_ewp_benchmark(
    tickers: list[str],
    test_start: str,
    test_end: str,
    interval: Interval,
    fee_rate: float,
) -> pd.DataFrame:
    """
    Generates an Equal-Weight Buy & Hold portfolio benchmark.

    The benchmark assumes an equal capital allocation (1/N) across all provided
    tickers at the start of the test period. Each asset receives the same initial
    investment and is then held without any subsequent rebalancing for the entire
    duration of the backtest.

    Key Methodological Assumptions:
    1. Initial Equal Allocation: Capital is split evenly across all assets at
       the beginning of the test period.
    2. Buy & Hold Strategy: No rebalancing occurs after the initial allocation.
       Asset weights are allowed to drift naturally according to their relative
       performance.
    3. Transaction Costs: Commissions are deducted at portfolio creation (entry)
       and upon final liquidation (exit).
    4. Delisting Handling: If an asset is delisted (missing data) during the
       test period, its last known valuation is frozen using forward-fill.
       This simulates a forced liquidation at the last available price (incurring
       an exit fee), with the recovered capital held as uninvested cash for the
       remainder of the backtest.
    """
    all_data = load_data(
        tickers=tickers,
        start=test_start,
        end=test_end,
        interval=interval,
    )

    invested_data = all_data.div(all_data.iloc[0]) * (1 - fee_rate)

    forward_filled = invested_data.ffill()
    is_delisted = all_data.isna()

    portfolio_values = invested_data.copy()
    portfolio_values[is_delisted] = forward_filled[is_delisted] * (1 - fee_rate)

    last_idx = portfolio_values.index[-1]
    active_assets = ~is_delisted.loc[last_idx]
    portfolio_values.loc[last_idx, active_assets] *= 1 - fee_rate

    portfolio_cum = portfolio_values.mean(axis=1)

    benchmark = pd.DataFrame(index=all_data.index)
    benchmark["ewp_return"] = portfolio_cum - 1.0
    benchmark["ewp_pct"] = portfolio_cum.pct_change()
    benchmark.loc[benchmark.index[0], "ewp_pct"] = 0.0

    return benchmark


def save_dataframe(
    df: pd.DataFrame, file_name: str, directory: str | Path = None
) -> None:
    if directory:
        target_dir = Path(directory)
    else:
        target_dir = get_project_root() / "results"
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / f"{file_name}.parquet"

    df_to_save = df

    if df.index.name is not None:
        df_to_save = df.reset_index()

    df_to_save.to_parquet(path, engine="pyarrow", index=False)


def save_strategy_result(
    result: StrategyResult, file_name: str, directory: str | Path = None
) -> None:
    if directory:
        target_dir = Path(directory)
    else:
        target_dir = get_project_root() / "results"
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / f"{file_name}.parquet"

    table = pa.Table.from_pandas(df=result.data)  # noqa
    metadata = {
        "ticker_x": result.ticker_x,
        "ticker_y": result.ticker_y,
        "start": result.start,
        "end": result.end,
        "interval": result.interval,
        "fee_rate": float(result.fee_rate),
        "stats_json": result.stats.to_json(),
        "exec_logger_json": result.exec_logger.to_json(),
    }

    custom_meta_key = "strategy_params".encode("utf-8")
    custom_meta_value = json.dumps(metadata).encode("utf-8")

    existing_meta = table.schema.metadata or {}
    new_meta = {**existing_meta, custom_meta_key: custom_meta_value}

    table = table.replace_schema_metadata(new_meta)
    pq.write_table(table, path)


def load_dataframe(file_name: str, directory: str | None = None) -> pd.DataFrame:
    PARQUET_DIR = get_project_root() / "results"
    if directory:
        path = PARQUET_DIR / f"{directory}/{file_name}.parquet"
    else:
        path = PARQUET_DIR / f"{file_name}.parquet"

    table = pq.read_table(path)
    df = table.to_pandas()

    return df


def load_strategy_result(
    file_name: str, directory: str | None = None
) -> StrategyResult:
    PARQUET_DIR = get_project_root() / "results"
    if directory:
        path = PARQUET_DIR / f"{directory}/{file_name}.parquet"
    else:
        path = PARQUET_DIR / f"{file_name}.parquet"

    table = pq.read_table(path)
    df = table.to_pandas()

    raw_meta = table.schema.metadata.get(b"strategy_params")
    meta = json.loads(raw_meta.decode("utf-8"))

    return StrategyResult(
        data=df,
        ticker_x=meta["ticker_x"],
        ticker_y=meta["ticker_y"],
        start=meta["start"],
        end=meta["end"],
        interval=meta["interval"],
        fee_rate=float(meta["fee_rate"]),
        stats=pd.read_json(StringIO(meta["stats_json"])),
        exec_logger=pd.read_json(StringIO(meta["exec_logger_json"])),
    )
