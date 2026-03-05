import calendar
import numpy as np
import pandas as pd

from modules.core.enums import Interval
from modules.data_services.data_utils import get_steps


def calculate_stats(
    df: pd.DataFrame,
    exec_log_df: pd.DataFrame,
    initial_cash: float,
    interval: Interval,
    risk_free_rate_annual: float,
) -> pd.DataFrame:
    """
    Calculates comprehensive strategy performance metrics combining:
    1. Time-series analysis (Sharpe, CAGR, Drawdown etc.) based on the equity curve.
    2. Trade execution analysis (Win Rate, Avg Trade etc.) based on the execution logger.
    """
    df = df.dropna(subset=["equity"]).copy()
    steps_per_day = get_steps(interval)

    if not df.empty:
        start_year = df.index[0].year
        last_ts = df.index[-1]
        end_year = last_ts.year

        if last_ts.month == 1 and last_ts.day == 1:
            end_year -= 1

        if end_year < start_year:
            end_year = start_year

        if start_year == end_year:
            days_in_year = 366 if calendar.isleap(start_year) else 365
        else:
            days_in_year = 365.25
    else:
        days_in_year = 365

    periods_per_year = steps_per_day * days_in_year

    def calculate_time_series_metrics(pnl_series: pd.Series) -> dict:
        """Calculates risk-adjusted return metrics based on the continuous equity curve."""
        pnl_series = pnl_series.iloc[1:]

        equity_curve = pnl_series + initial_cash
        returns = equity_curve.pct_change(fill_method=None).dropna()

        total_pnl = pnl_series.iloc[-1]
        total_return = total_pnl / initial_cash

        # Volatility
        period_volatility = returns.std() if not pd.isna(returns.std()) else None

        actual_periods = len(returns)
        annual_volatility = (
            period_volatility * np.sqrt(periods_per_year)
            if period_volatility is not None
            else None
        )

        # CAGR (Compound Annual Growth Rate)
        if len(equity_curve) > 0 and equity_curve.iloc[0] > 0:
            years = actual_periods / periods_per_year
            if years <= 0 or equity_curve.iloc[-1] <= 0:
                cagr = None
            else:
                cagr = (equity_curve.iloc[-1] / equity_curve.iloc[0]) ** (1 / years) - 1
        else:
            cagr = None

        period_rf = (1 + risk_free_rate_annual) ** (1 / periods_per_year) - 1

        # Sharpe ratio
        if period_volatility not in (None, 0, np.nan):
            sharpe_ratio = (returns.mean() - period_rf) / period_volatility
        else:
            sharpe_ratio = None

        sharpe_ratio_annual = (
            (cagr - risk_free_rate_annual) / annual_volatility
            if annual_volatility not in (0, None) and cagr is not None
            else None
        )

        # Sortino ratio
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std()

        if downside_std not in (None, 0, np.nan):
            sortino_ratio = (returns.mean() - period_rf) / downside_std
        else:
            sortino_ratio = None

        annual_downside_std = (
            downside_std * np.sqrt(periods_per_year)
            if downside_std not in (None, 0, np.nan)
            else None
        )

        sortino_ratio_annual = (
            (cagr - risk_free_rate_annual) / annual_downside_std
            if annual_downside_std not in (0, None)
            and annual_downside_std is not None
            and cagr is not None
            else None
        )

        # Maximum drawdown
        cumulative_max = equity_curve.cummax()
        drawdown = (equity_curve - cumulative_max) / cumulative_max
        max_drawdown = drawdown.min()

        # Calmar ratio
        calmar_ratio = total_return / abs(max_drawdown) if max_drawdown != 0 else None
        calmar_ratio_annual = (
            cagr / abs(max_drawdown) if max_drawdown != 0 and cagr is not None else None
        )

        return {
            "total_return": total_return,
            "cagr": cagr,
            "volatility": period_volatility,
            "volatility_annual": annual_volatility,
            "max_drawdown": max_drawdown,
            "sharpe_ratio": sharpe_ratio,
            "sharpe_ratio_annual": sharpe_ratio_annual,
            "sortino_ratio": sortino_ratio,
            "sortino_ratio_annual": sortino_ratio_annual,
            "calmar_ratio": calmar_ratio,
            "calmar_ratio_annual": calmar_ratio_annual,
        }

    def calculate_trade_metrics(is_net: bool) -> dict:
        """Calculates quantitative metrics based on closed positions from the execution log."""
        if exec_log_df.empty:
            return {
                "win_count": 0,
                "lose_count": 0,
                "win_rate": None,
                "avg_trade_duration": None,
                "max_win": None,
                "max_lose": None,
                "avg_win_return": None,
                "avg_lose_return": None,
                "avg_trade_return": None,
            }

        close_positions = exec_log_df[exec_log_df["position"] == 0.0].copy()

        if close_positions.empty:
            return {
                "win_count": 0,
                "lose_count": 0,
                "win_rate": None,
                "avg_trade_duration": None,
                "max_win": None,
                "max_lose": None,
                "avg_win_return": None,
                "avg_lose_return": None,
                "avg_trade_return": None,
            }

        if is_net:
            # Net: PnL - Fees
            fees = close_positions["fees"]
            trade_returns = (close_positions["pnl"] - fees) / close_positions[
                "entry_equity"
            ]
        else:
            # Gross: PnL only
            trade_returns = close_positions["pnl"] / close_positions["entry_equity"]

        wins = trade_returns[trade_returns >= 0]
        losses = trade_returns[trade_returns < 0]

        win_count = len(wins)
        lose_count = len(losses)
        total_trades = win_count + lose_count

        win_rate = win_count / total_trades if total_trades > 0 else None

        avg_trade_duration = (
            close_positions["time_in_pos"].mean() if total_trades > 0 else None
        )

        max_win = trade_returns.max() if not trade_returns.empty else None
        max_lose = trade_returns.min() if not trade_returns.empty else None

        avg_trade_ret = trade_returns.mean() if total_trades > 0 else None
        avg_win_ret = wins.mean() if win_count > 0 else None
        avg_lose_ret = losses.mean() if lose_count > 0 else None

        return {
            "win_count": win_count,
            "lose_count": lose_count,
            "win_rate": win_rate,
            "avg_trade_duration": avg_trade_duration,
            "max_win": max_win,
            "max_lose": max_lose,
            "avg_win_return": avg_win_ret,
            "avg_lose_return": avg_lose_ret,
            "avg_trade_return": avg_trade_ret,
        }

    gross_ts_stats = calculate_time_series_metrics(df["total_pnl"])
    net_ts_stats = calculate_time_series_metrics(df["total_net_pnl"])

    gross_trade_stats = calculate_trade_metrics(is_net=False)
    net_trade_stats = calculate_trade_metrics(is_net=True)

    gross_stats = {**gross_ts_stats, **gross_trade_stats}
    net_stats = {**net_ts_stats, **net_trade_stats}

    metrics_order = [
        "total_return",
        "cagr",
        "volatility",
        "volatility_annual",
        "max_drawdown",
        "win_count",
        "lose_count",
        "win_rate",
        "avg_trade_duration",
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
    ]

    stats_df = pd.DataFrame(
        {
            "metric": metrics_order,
            "gross": [gross_stats.get(m) for m in metrics_order],
            "net": [net_stats.get(m) for m in metrics_order],
        }
    ).set_index("metric")

    return stats_df.round(4)
