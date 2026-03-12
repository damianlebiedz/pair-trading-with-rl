from abc import ABC, abstractmethod
from typing import Literal, Union

import numpy as np
import pandas as pd


class ObjectiveScheme(ABC):
    @abstractmethod
    def calculate(
        self, stats: pd.DataFrame, metric_type: Literal["gross", "net"]
    ) -> float:
        pass


class TDASortino(ObjectiveScheme):
    """
    Time and Drawdown-Adjusted Sortino (TDA-Sortino).
    Custom objective function for pair trading.

    Formula:
    TDA_Sortino = (Sortino * sqrt(Trades)) / (sqrt(AvgTradeDuration) * (1 + lambda * MaxDD))
    """

    def __init__(
        self,
        drawdown_penalty_lambda: float = 10.0,
    ):
        self.lambda_penalty = drawdown_penalty_lambda

    def calculate(
        self,
        stats: Union[pd.DataFrame, dict],
        metric_type: Literal["gross", "net"] = "net",
    ) -> float:
        try:
            metrics = stats[metric_type]

            total_trades = metrics.get("win_count") + metrics.get("lose_count")
            sortino = metrics.get("sortino_annual_median")
            avg_duration = metrics.get("avg_trade_duration")
            max_dd = metrics.get("max_drawdown")

            if sortino is None or total_trades == 0 or avg_duration == 0 or sortino < 0:
                return None

            numerator = sortino * np.sqrt(total_trades)
            denominator = np.sqrt(avg_duration) * (
                1.0 + self.lambda_penalty * abs(max_dd)
            )

            tda_sortino = numerator / denominator

            return float(tda_sortino)

        except KeyError as e:
            print(f"Error calculating TDA-Sortino objective - missing key: {e}")
            return None

        except Exception as e:
            print(f"Unexpected error calculating TDA-Sortino: {e}")
            return None
