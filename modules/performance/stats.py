from typing import Literal
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from modules.data_services.data_utils import get_steps


def calculate_stats(
    df: pd.DataFrame,
    initial_cash: float,
    interval: Literal["1d", "4h", "1h", "30m", "15m", "5m", "3m", "1m"],
    risk_free_rate_annual: float,
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

        def equity_slope_r2(eq_curve: pd.Series) -> tuple[float | None, float | None]:
            if len(eq_curve) < 2:
                return None, None

            eq = eq_curve[eq_curve > 0]
            if len(eq) < 2:
                return None, None

            y = np.log(eq.values)
            x = np.arange(len(y))

            s, intercept = np.polyfit(x, y, 1)

            y_hat = s * x + intercept
            ss_res = np.sum((y - y_hat) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)

            r_2 = 1.0 - float(ss_res) / float(ss_tot) if ss_tot != 0 else None

            return s, r_2

        # Equity slope and R^2
        slope, r2 = equity_slope_r2(equity_curve)

        # Objective - Robust Median-Log-Sortino with R2
        def objective():
            """
            Wzór:
            Objective = (Median(ln(1+r)) / Downside_Deviation(ln(1+r))) * R2_log * ln(N)

            Gdzie:
            - Median(ln(1+r)): Typowy zysk ztransponowany na skalę logarytmiczną (wycina fuksy).
            - Downside_Deviation: Odchylenie strat (Sortino) na log-zwrotach.
            - R2_log: Liniowość wzrostu kapitału po logarytmowaniu.
            - ln(N): Premia za liczbę transakcji (wiarygodność statystyczna).
            """

            if total_trades < 15:
                return -1e2  # Kara za zbyt małą próbę statystyczną

            # 1. Przygotowanie log-zwrotów (Neutralizacja outlierów i fuksów)
            # Clip chroni przed logarytmowaniem wartości <= -1.0
            safe_returns = np.clip(trade_pnl / initial_cash, -0.99, None)
            log_returns = np.log(safe_returns + 1.0)

            # 2. Mediana Log-Zwrotów (Licznik - typowa efektywność)
            median_log_ret = np.median(log_returns)

            # 3. Downside Deviation (Mianownik - ryzyko strat)
            # Obliczamy zmienność tylko dla wyników ujemnych
            losses = log_returns[log_returns < 0]
            if len(losses) > 0:
                # Standardowa formuła Sortino: pierwiastek średniej kwadratów strat
                downside_dev = np.sqrt(np.mean(losses ** 2))
            else:
                downside_dev = 1e-6  # Idealny przypadek: brak strat w próbce

            # 4. Logarytmiczne R^2 (Liniowość / Powtarzalność)
            log_equity_curve = np.cumsum(log_returns)

            def get_log_r2(le_curve):
                # R2 wymaga zmienności; jeśli kapitał stoi w miejscu, R2 = 0
                if len(le_curve) < 5 or np.all(le_curve == le_curve[0]):
                    return 0.0
                y = le_curve
                x = np.arange(len(y)).reshape(-1, 1)
                lr: LinearRegression = LinearRegression().fit(x, y)
                return float(lr.score(x, y))

            r2_log = get_log_r2(log_equity_curve)

            # Medianowe Sortino * Liniowość * Logarytmiczna liczba tradów
            median_sortino = median_log_ret / (downside_dev + 1e-6)

            obj = median_sortino * r2_log * np.log(total_trades)

            return obj

        objective = objective()

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
            "r2": r2,
            "slope": slope,
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
        "r2",
        "slope",
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


def calculate_multi_pair_stats():
    ...
