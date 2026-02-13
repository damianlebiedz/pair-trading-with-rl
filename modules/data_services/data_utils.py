from functools import reduce
from io import StringIO
from pathlib import Path
from typing import Literal
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import json

from modules.performance.models import StrategyResult
from modules.data_services.data_loaders import load_data, get_project_root


def get_steps(
    interval: Literal["1d", "4h", "1h", "30m", "15m", "5m", "3m", "1m"],
) -> int:
    """Get steps of the interval."""
    if interval == "1d":
        return 1
    elif interval == "4h":
        return 6
    elif interval == "1h":
        return 24
    elif interval == "30m":
        return 48
    elif interval == "15m":
        return 96
    elif interval == "5m":
        return 288
    elif interval == "3m":
        return 480
    elif interval == "1m":
        return 1440
    else:
        raise ValueError(
            f"Wrong interval '{interval}', should be one of: '1d', '4h', '1h', '30m', '15m', '5m', '3m', '1m'."
        )


def add_log_prices(df: pd.DataFrame, ticker_x: str, ticker_y: str) -> None:
    """Add log prices to DataFrame."""
    df[f"{ticker_x}_log"] = np.log(df[ticker_x])
    df[f"{ticker_y}_log"] = np.log(df[ticker_y])


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
