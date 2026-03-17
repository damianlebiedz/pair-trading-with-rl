import pandas as pd

from modules.performance.models import StrategyResult


def stitch_strategy_results(
    results: list[StrategyResult],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Stitches sequential strategy results into a continuous timeline (Horizontal/Chronological Merge).

    This function is designed for Multi-Period analysis (e.g., Walk-Forward Optimization)
    where the simulation is executed in distinct, consecutive time chunks. It ensures
    that the equity curve remains continuous by applying the accumulated PnL/Return
    from the previous period as an offset to the next.

    Key operations:
    - Preserves chronological order of the provided results.
    - Adjusts cumulative columns (`total_pnl`, `total_net_pnl`, `total_return`,
      `total_net_return`) so that period N starts with the final values of period N-1.
    - Concatenates execution logs to form a complete trading history.

    Args:
        results (list[StrategyResult]): A list of strategy results ordered chronologically
            (e.g., Period 1, Period 2, ...).

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]:
            1. merged_df: A continuous time-series DataFrame with adjusted cumulative metrics.
            2. merged_exec_log_df: A combined execution log containing trades from all
               periods, sorted by time.

    Raises:
        ValueError: If the `results` list is empty.
    """
    if not results:
        raise ValueError("No results to stitch")

    merged_dfs = []
    exec_dfs = []

    cumulative_cols = [
        "total_pnl",
        "total_net_pnl",
        "total_return",
        "total_net_return",
        "total_fees",
    ]

    offsets = {col: 0.0 for col in cumulative_cols}

    for res in results:
        df = res.data.dropna(subset=["equity"]).copy()

        if "open_time" in df.columns:
            df = df.set_index("open_time")

        for col in cumulative_cols:
            if col in df.columns:
                df[col] += offsets[col]

        if "equity" in df.columns:
            df["equity"] += offsets.get("total_net_pnl", 0.0)

        merged_dfs.append(df)

        if not res.exec_logger.empty:
            temp_exec_df = res.exec_logger.copy()

            exec_dfs.append(temp_exec_df)

        if not df.empty:
            for col in cumulative_cols:
                if col in df.columns:
                    offsets[col] = df[col].iloc[-1]

    final_df = pd.concat(merged_dfs).sort_index()
    final_df = final_df[~final_df.index.duplicated(keep="first")]

    if exec_dfs:
        final_exec_df = (
            pd.concat(exec_dfs).sort_values(by="open_time").reset_index(drop=True)
        )
    else:
        final_exec_df = pd.DataFrame()

    return final_df, final_exec_df


def aggregate_strategy_results(
    results: list[StrategyResult],
    initial_cash: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Aggregates multiple simultaneous strategy results into a combined portfolio view (Vertical Merge).

    This function is designed for Multi-Pair analysis where multiple pairs are traded
    simultaneously over the same time index. It sums up the absolute PnL and Net PnL
    from all provided results to form a total portfolio performance curve.

    Key operations:
    - Sums `total_return` and `net_return` (absolute values) across all strategies.
    - Calculates aggregate percentage returns based on `total_initial_cash`.
    - Averages market exposure (`in_position`).
    - Concatenates execution logs from all pairs into a single timeline.

    Args:
        results (list[StrategyResult]): A list of strategy results from different pairs
            covering the same time range.
        initial_cash (float): The total initial capital allocated to the entire
            portfolio (denominator for aggregate percentage calculations).

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]:
            1. merged_df: A DataFrame with aggregated time-series metrics (summed PnL,
               averaged position exposure).
            2. merged_exec_log_df: A combined execution log containing trades from all
               strategies, sorted by time.

    Raises:
        ValueError: If the `results` list is empty.
    """
    if not results:
        raise ValueError("No results to aggregate")

    base_df = results[0].data.dropna(subset=["equity"])
    base_index = base_df.index

    total_pnl_sum = pd.Series(0.0, index=base_index)
    net_pnl_sum = pd.Series(0.0, index=base_index)
    position_sum = pd.Series(0.0, index=base_index)
    total_fees_sum = pd.Series(0.0, index=base_index)

    exec_dfs = []

    for res in results:
        df = res.data.reindex(base_index).infer_objects(copy=False)

        total_pnl_sum += df["total_pnl"]
        net_pnl_sum += df["total_net_pnl"]
        position_sum += df["position"].abs()

        if "total_fees" in df.columns:
            total_fees_sum += df["total_fees"].fillna(0.0)

        if not res.exec_logger.empty:
            temp_exec_df = res.exec_logger.copy()
            exec_dfs.append(temp_exec_df)

    merged_df = pd.DataFrame(index=base_index)

    merged_df["total_pnl"] = total_pnl_sum
    merged_df["total_net_pnl"] = net_pnl_sum
    merged_df["equity"] = initial_cash + merged_df["total_net_pnl"]
    merged_df["total_fees"] = total_fees_sum
    merged_df["in_position"] = position_sum / len(results)

    merged_df["total_return"] = merged_df["total_pnl"] / initial_cash
    merged_df["total_net_return"] = merged_df["total_net_pnl"] / initial_cash

    if exec_dfs:
        merged_exec_log_df = (
            pd.concat(exec_dfs).sort_values(by="open_time").reset_index(drop=True)
        )
    else:
        merged_exec_log_df = pd.DataFrame()

    return merged_df, merged_exec_log_df
