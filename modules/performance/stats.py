from typing import Literal
import numpy as np
import pandas as pd

from modules.core.models import StrategyResult
from modules.data_services.data_utils import get_steps


def calculate_stats(
    df: pd.DataFrame,
    initial_cash: float,
    interval: Literal["1d", "4h", "1h", "30m", "15m", "5m", "3m", "1m"],
    risk_free_rate_annual: float,
    min_trades_per_pair: int,
) -> pd.DataFrame:
    steps_per_day = get_steps(interval)
    periods_per_year = steps_per_day * 365

    def calc_trade_array(
        pnl_series: pd.Series, position_series: pd.Series
    ) -> np.ndarray:
        prev = 0
        open_idx = None
        trade_pnl = []

        for i in range(len(pnl_series)):
            pos = position_series.iloc[i]

            if prev == 0 and pos != 0:
                open_idx = i
            elif (
                (prev < 0 <= pos)
                or (prev > 0 >= pos)
                or (prev != 0 and i == len(position_series) - 1)
            ):
                if open_idx is not None:
                    pnl = pnl_series.iloc[i] - pnl_series.iloc[open_idx]
                    trade_pnl.append(pnl)
                open_idx = i if pos != 0 else None
            prev = pos

        return np.array(trade_pnl)

    def compute_stats(pnl_series: pd.Series) -> dict:
        equity_curve = pnl_series + initial_cash
        returns = equity_curve.pct_change().dropna()

        total_pnl = pnl_series.iloc[-1]
        total_return = total_pnl / initial_cash

        pnl_series, position_series = pnl_series.align(df["position"], join="inner")
        trade_pnl = calc_trade_array(pnl_series, position_series)

        # Total wins / Total losses
        total_wins = int(np.sum(trade_pnl > 0))
        total_losses = int(np.sum(trade_pnl < 0))

        # Win rate
        total_trades = total_wins + total_losses
        win_rate = total_wins / total_trades if total_trades > 0 else None

        # Max win / Max lose
        winning_trades = trade_pnl[trade_pnl > 0]
        losing_trades = trade_pnl[trade_pnl < 0]
        max_win_pct = (
            winning_trades.max() / initial_cash if len(winning_trades) > 0 else None
        )
        max_lose_pct = (
            losing_trades.min() / initial_cash if len(losing_trades) > 0 else None
        )

        # Avg win / Avg lose / Avg trade return
        avg_win_trade_pct = (
            winning_trades.mean() / initial_cash if total_wins > 0 else None
        )
        avg_lose_trade_pct = (
            losing_trades.mean() / initial_cash if total_losses > 0 else None
        )
        avg_trade_ret_pct = (
            np.mean(trade_pnl) / initial_cash if total_trades > 0 else None
        )

        # Volatility
        period_volatility = returns.std() if not pd.isna(returns.std()) else None
        annual_volatility = (
            period_volatility * np.sqrt(periods_per_year)
            if period_volatility is not None
            else None
        )

        # CAGR (Compound Annual Growth Rate)
        if len(equity_curve) > 0 and equity_curve.iloc[0] > 0:
            years = len(equity_curve) / periods_per_year
            if years <= 0 or equity_curve.iloc[-1] <= 0:
                cagr = None
            else:
                cagr = (equity_curve.iloc[-1] / equity_curve.iloc[0]) ** (1 / years) - 1
        else:
            cagr = None

        # Sharpe ratio
        period_rf = (1 + risk_free_rate_annual) ** (1 / periods_per_year) - 1
        if period_volatility not in (None, 0, np.nan):
            sharpe_ratio = (returns.mean() - period_rf) / period_volatility
        else:
            sharpe_ratio = None
        sharpe_ratio_annual = (
            (cagr - risk_free_rate_annual) / annual_volatility
            if annual_volatility not in (0, None)
            else None
        )

        # Sortino ratio
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std()
        sortino_ratio = (
            returns.mean() / downside_std
            if downside_std not in (None, 0, np.nan)
            else None
        )
        sortino_ratio_annual = (
            sortino_ratio * np.sqrt(periods_per_year)
            if sortino_ratio is not None
            else None
        )

        # Maximum drawdown
        cumulative_max = equity_curve.cummax()
        drawdown = (equity_curve - cumulative_max) / cumulative_max
        max_drawdown = drawdown.min()

        # Calmar ratio
        calmar_ratio = total_return / abs(max_drawdown) if max_drawdown != 0 else None
        calmar_ratio_annual = cagr / abs(max_drawdown) if max_drawdown != 0 else None

        # Objective
        objective = -100 if total_trades < min_trades_per_pair else sortino_ratio_annual

        return {
            "total_return": total_return,
            "cagr": cagr,
            "volatility": period_volatility,
            "volatility_annual": annual_volatility,
            "max_drawdown": max_drawdown,
            "win_count": total_wins,
            "lose_count": total_losses,
            "win_rate": win_rate,
            "max_win": max_win_pct,
            "max_lose": max_lose_pct,
            "avg_win_return": avg_win_trade_pct,
            "avg_lose_return": avg_lose_trade_pct,
            "avg_trade_return": avg_trade_ret_pct,
            "sharpe_ratio": sharpe_ratio,
            "sharpe_ratio_annual": sharpe_ratio_annual,
            "sortino_ratio": sortino_ratio,
            "sortino_ratio_annual": sortino_ratio_annual,
            "calmar_ratio": calmar_ratio,
            "calmar_ratio_annual": calmar_ratio_annual,
            "objective": objective,
        }

    gross_stats = compute_stats(df["total_return"])
    net_stats = compute_stats(df["net_return"])

    metrics_order = [
        "total_return",
        "cagr",
        "volatility",
        "volatility_annual",
        "max_drawdown",
        "win_count",
        "lose_count",
        "win_rate",
        "max_win",
        "max_lose",
        "avg_win_return",
        "avg_lose_return",
        "avg_trade_return",
        "sharpe_ratio",
        "sharpe_ratio_annual",
        "sortino_ratio",
        "sortino_ratio_annual",
        "calmar_ratio",
        "calmar_ratio_annual",
        "objective",
    ]

    stats_df = pd.DataFrame(
        {
            "metric": metrics_order,
            "gross": [gross_stats[m] for m in metrics_order],
            "net": [net_stats[m] for m in metrics_order],
        }
    ).set_index("metric")

    return stats_df.round(4)


def calculate_multi_pair_stats(
    merged_df: pd.DataFrame,
    individual_stats_dfs: list[pd.DataFrame],
    total_initial_cash: float,
    interval: str,
    risk_free_rate_annual: float,
    number_of_pairs: int,
    min_trades_per_pair: int,
) -> pd.DataFrame:
    """
    Calculates statistics for a multi-pair portfolio.
    Hybrid approach:
    - Time-series metrics (Sharpe, DD, CAGR) are calculated on the merged equity curve.
    - Trade metrics (Win rate, Counts) are aggregated from individual results.
    """

    merged_df_for_calc = merged_df.copy()
    if "position" not in merged_df_for_calc.columns:
        merged_df_for_calc["position"] = 0

    portfolio_stats = calculate_stats(
        df=merged_df_for_calc,
        initial_cash=total_initial_cash,
        interval=interval,
        risk_free_rate_annual=risk_free_rate_annual,
        min_trades_per_pair=min_trades_per_pair,
    )

    agg_stats = {"gross": {}, "net": {}}

    min_total_trades = min_trades_per_pair * number_of_pairs

    for col in ["gross", "net"]:
        ind_series = [stats_df[col] for stats_df in individual_stats_dfs]

        total_wins = sum(s["win_count"] for s in ind_series)
        total_losses = sum(s["lose_count"] for s in ind_series)
        total_trades = total_wins + total_losses

        win_rate = total_wins / total_trades if total_trades > 0 else None

        avg_win_return = (
            np.mean([s["avg_win_return"] for s in ind_series]) / number_of_pairs
        )
        avg_lose_return = (
            np.mean([s["avg_lose_return"] for s in ind_series]) / number_of_pairs
        )
        avg_trade_return = (
            np.mean([s["avg_trade_return"] for s in ind_series]) / number_of_pairs
        )
        max_win = np.max([s["max_win"] for s in ind_series]) / number_of_pairs
        max_lose = np.min([s["max_lose"] for s in ind_series]) / number_of_pairs

        current_stats = portfolio_stats[col].to_dict()
        raw_objective = current_stats["sortino_ratio_annual"]

        if total_trades < min_total_trades:
            objective = -100.0
        else:
            objective = raw_objective

        current_stats.update(
            {
                "win_count": total_wins,
                "lose_count": total_losses,
                "win_rate": win_rate,
                "avg_win_return": avg_win_return,
                "avg_lose_return": avg_lose_return,
                "avg_trade_return": avg_trade_return,
                "max_win": max_win,
                "max_lose": max_lose,
                "objective": objective,
            }
        )

        agg_stats[col] = current_stats

    metrics_order = portfolio_stats.index

    final_df = pd.DataFrame(
        {
            "metric": metrics_order,
            "gross": [agg_stats["gross"].get(m) for m in metrics_order],
            "net": [agg_stats["net"].get(m) for m in metrics_order],
        }
    ).set_index("metric")

    return final_df.round(4)


def aggregate_strategy_results(
    results: list[StrategyResult], total_initial_cash: float
) -> pd.DataFrame:
    if not results:
        raise ValueError("No results to aggregate")

    base_df = results[0].data.copy()
    base_index = base_df.index

    total_return_sum = pd.Series(0.0, index=base_index)
    net_return_sum = pd.Series(0.0, index=base_index)
    fees_sum = pd.Series(0.0, index=base_index)

    for res in results:
        df = res.data
        total_return_sum = total_return_sum.add(df["total_return"], fill_value=0)
        net_return_sum = net_return_sum.add(df["net_return"], fill_value=0)
        if "fees" in df.columns:
            fees_sum = fees_sum.add(df["fees"], fill_value=0)

    merged_df = pd.DataFrame(index=base_index)
    merged_df["total_return"] = total_return_sum
    merged_df["net_return"] = net_return_sum
    merged_df["fees"] = fees_sum

    merged_df["total_return_pct"] = merged_df["total_return"] / total_initial_cash
    merged_df["net_return_pct"] = merged_df["net_return"] / total_initial_cash

    merged_df["position"] = 0
    merged_df["z_score"] = 0
    merged_df["entry_thr"] = 0
    merged_df["exit_thr"] = 0

    return merged_df
