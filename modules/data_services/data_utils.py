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
    df[f"{ticker_x}_{Source.LOG}"] = np.log(df[ticker_x])
    df[f"{ticker_y}_{Source.LOG}"] = np.log(df[ticker_y])


def merge_by_pair(dfs: list[pd.DataFrame], keep_cols: list[list[str]]) -> pd.DataFrame:
    """Merge dataframes from statistical tests into one dataframe."""
    trimmed = []
    for df, cols in zip(dfs, keep_cols):
        trimmed.append(df[["pair"] + cols])

    merged = reduce(
        lambda left, right: pd.merge(left, right, on="pair", how="outer"), trimmed
    )
    return merged


def load_btc_benchmark(test_start: str, test_end: str, interval: str) -> pd.DataFrame:
    btc_data = load_data(
        tickers=["BTCUSDT"],
        start=test_start,
        end=test_end,
        interval=interval,
    )
    btc_data["BTC_pct"] = btc_data["BTCUSDT"].pct_change()
    btc_data.loc[btc_data.index[0], "BTC_pct"] = 0.0
    btc_data["BTC_return"] = (1 + btc_data["BTC_pct"]).cumprod() - 1

    return btc_data


def load_ewp_benchmark(
    tickers: list[str], test_start: str, test_end: str, interval: Interval
) -> pd.DataFrame:
    """
    Generates an Equally Weighted (EW) portfolio benchmark with continuous rebalancing.

    The strategy assumes an equal capital allocation (1/N) across all provided tickers.
    By utilizing percentage returns, the benchmark remains invariant to the nominal
    prices of the underlying assets, ensuring that high-priced assets (e.g., BTC)
    do not disproportionately influence the index compared to lower-priced assets.

    Key Methodological Assumptions:
    1. Continuous Rebalancing: The portfolio is rebalanced to equal weights at every
       specified interval (e.g., 1h). This effectively simulates selling outperformers
       and buying underperformers to maintain the 1/N distribution at each step.
    2. Zero Transaction Costs: This benchmark represents a theoretical "frictionless"
       market return. It does not account for trading commissions, bid-ask spreads,
       or execution slippage.
    3. Arithmetic Mean Returns: The portfolio return for each period is calculated
       as the simple arithmetic average of the individual asset returns.

    Calculations:
    - Computes period-over-period percentage changes for all assets.
    - Derives the aggregate portfolio return per interval.
    - Generates a cumulative return series (Equity Curve) starting from zero.

    This serves as a passive multi-asset baseline to evaluate the Alpha generated
    by the active strategy over a simple buy-and-hold-weighted index.
    """
    all_data = load_data(
        tickers=tickers,
        start=test_start,
        end=test_end,
        interval=interval,
    )
    returns_df = all_data.pct_change()
    portfolio_benchmark = pd.DataFrame(index=all_data.index)
    portfolio_benchmark["portfolio_pct"] = returns_df.mean(axis=1)
    portfolio_benchmark.loc[portfolio_benchmark.index[0], "portfolio_pct"] = 0.0
    portfolio_benchmark["ewp_return"] = (
        1 + portfolio_benchmark["portfolio_pct"]
    ).cumprod() - 1

    return portfolio_benchmark


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
