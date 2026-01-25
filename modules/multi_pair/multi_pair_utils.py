import pandas as pd

from modules.core.models import StrategyResult


def aggregate_strategy_results(
    results: list[StrategyResult], total_initial_cash: float,
) -> pd.DataFrame:
    if not results:
        raise ValueError("No results to aggregate")

    base_df = results[0].data
    base_index = base_df.index

    total_return_sum = pd.Series(0.0, index=base_index)
    net_return_sum = pd.Series(0.0, index=base_index)
    position_sum = pd.Series(0.0, index=base_index)

    for res in results:
        df = res.data.reindex(base_index).fillna(0)

        total_return_sum += df["total_return"]
        net_return_sum += df["net_return"]
        position_sum += df["position"].abs()

    merged_df = pd.DataFrame(index=base_index)
    merged_df["total_return"] = total_return_sum
    merged_df["net_return"] = net_return_sum
    merged_df["in_position"] = position_sum / len(results)

    merged_df["total_return_pct"] = merged_df["total_return"] / total_initial_cash
    merged_df["net_return_pct"] = merged_df["net_return"] / total_initial_cash

    return merged_df
