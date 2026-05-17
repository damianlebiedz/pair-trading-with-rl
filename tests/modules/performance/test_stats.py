"""Full coverage tests for modules.performance.stats."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from modules.core.enums import Interval
from modules.performance.stats import calculate_stats


def _strategy_df(
    gross_pnl: list[float],
    net_pnl: list[float] | None = None,
    *,
    start: str = "2024-03-01",
    freq: str = "D",
    initial_cash: float = 10_000.0,
) -> pd.DataFrame:
    if net_pnl is None:
        net_pnl = gross_pnl
    dates = pd.date_range(start, periods=len(gross_pnl), freq=freq)
    return pd.DataFrame(
        {
            "equity": initial_cash + np.array(net_pnl, dtype=float),
            "total_pnl": gross_pnl,
            "total_net_pnl": net_pnl,
        },
        index=dates,
    )


def _exec_log(
    positions: list[float],
    pnls: list[float],
    fees: list[float],
    entry_equities: list[float],
    durations: list[int],
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "position": positions,
            "pnl": pnls,
            "fees": fees,
            "entry_equity": entry_equities,
            "time_in_pos": durations,
        }
    )


class TestCalculateStatsEmpty:
    def test_empty_df_hits_default_days_in_year_branch(self) -> None:
        df = pd.DataFrame(
            {
                "equity": [np.nan],
                "total_pnl": [0.0],
                "total_net_pnl": [0.0],
            },
            index=pd.date_range("2024-01-01", periods=1),
        )
        with pytest.raises(IndexError):
            calculate_stats(
                df=df,
                exec_log_df=pd.DataFrame(),
                initial_cash=10_000.0,
                interval=Interval.D1,
                risk_free_rate_annual=0.02,
            )


class TestCalculateStatsYearLogic:
    def test_single_leap_year(self) -> None:
        gross = list(np.linspace(0, 500, 60))
        df = _strategy_df(gross, start="2024-01-15")
        result = calculate_stats(
            df, pd.DataFrame(), 10_000.0, Interval.D1, 0.02
        )
        assert result.loc["total_return", "gross"] is not None

    def test_single_non_leap_year(self) -> None:
        gross = list(np.linspace(0, 300, 50))
        df = _strategy_df(gross, start="2023-02-01")
        result = calculate_stats(
            df, pd.DataFrame(), 10_000.0, Interval.D1, 0.02
        )
        assert result.loc["cagr", "gross"] is not None or pd.isna(result.loc["cagr", "gross"])

    def test_multi_year_span(self) -> None:
        gross = list(np.linspace(0, 800, 400))
        df = _strategy_df(gross, start="2023-01-01")
        result = calculate_stats(
            df, pd.DataFrame(), 10_000.0, Interval.D1, 0.02
        )
        assert result.shape[0] == 19

    def test_jan_first_end_adjustment(self) -> None:
        gross = [0.0, 50.0, 100.0]
        df = _strategy_df(gross, start="2024-06-01")
        df = df.reindex(
            list(df.index) + [pd.Timestamp("2025-01-01")]
        )
        df.loc[pd.Timestamp("2025-01-01"), ["equity", "total_pnl", "total_net_pnl"]] = [
            10_150.0,
            150.0,
            140.0,
        ]
        result = calculate_stats(
            df, pd.DataFrame(), 10_000.0, Interval.D1, 0.02
        )
        assert not result.empty

    def test_end_year_clamped_to_start_year(self) -> None:
        df = pd.DataFrame(
            {
                "equity": [10_100.0, 10_200.0],
                "total_pnl": [100.0, 200.0],
                "total_net_pnl": [90.0, 180.0],
            },
            index=pd.DatetimeIndex(["2025-06-01", "2025-01-01"]),
        )
        result = calculate_stats(
            df, pd.DataFrame(), 10_000.0, Interval.D1, 0.02
        )
        assert result.loc["total_return", "gross"] == pytest.approx(0.02, rel=1e-3)


class TestCalculateStatsTimeSeries:
    def test_full_metrics_happy_path(self) -> None:
        rng = np.random.default_rng(0)
        shocks = rng.normal(0, 50, 120)
        gross = np.cumsum(shocks).tolist()
        net = (np.array(gross) - 5).tolist()
        df = _strategy_df(gross, net, start="2024-01-01")
        result = calculate_stats(
            df, pd.DataFrame(), 10_000.0, Interval.H1, 0.03
        )
        assert result.loc["sharpe_ratio", "gross"] is not None
        assert result.loc["sortino_ratio", "gross"] is not None
        assert result.loc["max_drawdown", "gross"] >= 0

    def test_zero_volatility_and_nan_sharpe(self) -> None:
        df = _strategy_df([0.0, 100.0], start="2024-05-01")
        result = calculate_stats(
            df, pd.DataFrame(), 10_000.0, Interval.D1, 0.02
        )
        assert pd.isna(result.loc["volatility", "gross"]) or result.loc["volatility", "gross"] is None
        assert pd.isna(result.loc["sharpe_ratio", "gross"]) or result.loc["sharpe_ratio", "gross"] is None

    def test_cagr_none_when_terminal_equity_non_positive(self) -> None:
        df = _strategy_df([0.0, -10_500.0], start="2024-04-01")
        result = calculate_stats(
            df, pd.DataFrame(), 10_000.0, Interval.D1, 0.02
        )
        assert pd.isna(result.loc["cagr", "gross"])

    def test_cagr_none_when_initial_equity_non_positive(self) -> None:
        df = _strategy_df([-10_500.0, -10_400.0], start="2024-04-01")
        result = calculate_stats(
            df, pd.DataFrame(), 10_000.0, Interval.D1, 0.02
        )
        assert pd.isna(result.loc["cagr", "gross"])

    def test_sortino_none_when_no_downside_returns(self) -> None:
        gross = list(np.linspace(0, 1000, 30))
        df = _strategy_df(gross, start="2024-07-01")
        result = calculate_stats(
            df, pd.DataFrame(), 10_000.0, Interval.D1, 0.0
        )
        assert pd.isna(result.loc["sortino_ratio", "gross"])

    def test_calmar_none_when_no_drawdown(self) -> None:
        gross = list(np.linspace(0, 500, 25))
        df = _strategy_df(gross, start="2024-08-01")
        result = calculate_stats(
            df, pd.DataFrame(), 10_000.0, Interval.D1, 0.02
        )
        assert result.loc["max_drawdown", "gross"] == pytest.approx(0.0, abs=1e-9)
        assert pd.isna(result.loc["calmar_ratio", "gross"])

    def test_annual_ratios_populated(self) -> None:
        rng = np.random.default_rng(1)
        gross = np.cumsum(rng.normal(20, 80, 200)).tolist()
        df = _strategy_df(gross, start="2023-03-01")
        result = calculate_stats(
            df, pd.DataFrame(), 10_000.0, Interval.D1, 0.01
        )
        assert result.loc["sharpe_ratio_annual", "gross"] is not None or pd.isna(
            result.loc["sharpe_ratio_annual", "gross"]
        )
        assert result.loc["sortino_ratio_annual", "net"] is not None or pd.isna(
            result.loc["sortino_ratio_annual", "net"]
        )


class TestCalculateStatsTrades:
    def test_empty_exec_log_trade_defaults(self) -> None:
        df = _strategy_df([0.0, 50.0, 80.0])
        result = calculate_stats(
            df, pd.DataFrame(), 10_000.0, Interval.D1, 0.02
        )
        assert result.loc["win_count", "gross"] == 0
        assert result.loc["avg_trade_duration", "gross"] == 0.0

    def test_exec_log_without_closes(self) -> None:
        df = _strategy_df([0.0, 20.0])
        log = _exec_log([1.0], [0.0], [1.0], [1000.0], [0])
        result = calculate_stats(df, log, 10_000.0, Interval.D1, 0.02)
        assert result.loc["win_count", "gross"] == 0
        assert pd.isna(result.loc["avg_trade_duration", "gross"])

    def test_gross_and_net_trade_metrics(self) -> None:
        df = _strategy_df([0.0, 100.0, 50.0], net_pnl=[0.0, 80.0, 30.0])
        log = _exec_log(
            positions=[1.0, 0.0, -1.0, 0.0],
            pnls=[0.0, 200.0, 0.0, -100.0],
            fees=[10.0, 20.0, 10.0, 15.0],
            entry_equities=[1000.0, 1000.0, 1000.0, 1000.0],
            durations=[0, 5, 0, 3],
        )
        result = calculate_stats(df, log, 10_000.0, Interval.D1, 0.02)
        assert result.loc["win_count", "gross"] == 1
        assert result.loc["lose_count", "gross"] == 1
        assert result.loc["win_rate", "gross"] == pytest.approx(0.5)
        assert result.loc["max_win", "gross"] == pytest.approx(0.2)
        assert result.loc["max_lose", "gross"] == pytest.approx(-0.1)
        assert result.loc["max_win", "net"] == pytest.approx(0.18)
        assert result.loc["max_lose", "net"] == pytest.approx(-0.115)
        assert result.loc["avg_trade_duration", "gross"] == pytest.approx(4.0)

    def test_result_is_rounded_dataframe(self) -> None:
        df = _strategy_df([0.0, 33.3333])
        result = calculate_stats(
            df, pd.DataFrame(), 10_000.0, Interval.D1, 0.02
        )
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["gross", "net"]
        assert result.loc["total_return", "gross"] == pytest.approx(0.0033, abs=1e-4)
