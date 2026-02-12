import pandas as pd

from modules.core.models import StrategyResult

pd.set_option("future.no_silent_downcasting", True)


def aggregate_strategy_results(
    results: list[StrategyResult],
    total_initial_cash: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not results:
        raise ValueError("No results to aggregate")

    base_df = results[0].data
    base_index = base_df.index

    total_return_sum = pd.Series(0.0, index=base_index)
    net_return_sum = pd.Series(0.0, index=base_index)
    position_sum = pd.Series(0.0, index=base_index)

    exec_dfs = []

    for res in results:
        df = res.data.reindex(base_index).fillna(0)

        total_return_sum += df["total_return"]
        net_return_sum += df["net_return"]
        position_sum += df["position"].abs()

        if not res.exec_logger.empty:
            temp_exec_df = res.exec_logger.copy()
            exec_dfs.append(temp_exec_df)

    merged_df = pd.DataFrame(index=base_index)
    merged_df["total_return"] = total_return_sum
    merged_df["net_return"] = net_return_sum
    merged_df["in_position"] = position_sum / len(results)

    merged_df["total_return_pct"] = merged_df["total_return"] / total_initial_cash
    merged_df["net_return_pct"] = merged_df["net_return"] / total_initial_cash

    merged_exec_log_df = (
        pd.concat(exec_dfs).sort_values(by="open_time").reset_index(drop=True)
    )

    return merged_df, merged_exec_log_df
