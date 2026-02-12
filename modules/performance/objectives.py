from abc import ABC, abstractmethod
from typing import Literal
import pandas as pd


class ObjectiveScheme(ABC):
    @abstractmethod
    def calculate(
        self, stats: pd.DataFrame, metric_type: Literal["gross", "net"]
    ) -> float:
        pass


class SortinoWithPenalty(ObjectiveScheme):
    def __init__(self, min_trades_per_pair: int, penalty_value: float = -100.0):
        self.min_trades = min_trades_per_pair
        self.penalty_value = penalty_value

    def calculate(self, stats: pd.DataFrame, metric_type: str = "net") -> float:
        try:
            metrics = stats[metric_type]

            total_trades = metrics.get("win_count", 0) + metrics.get("lose_count", 0)
            sortino = metrics.get("sortino_ratio_annual")

            if total_trades < self.min_trades or sortino is None:
                return self.penalty_value

            return float(sortino)

        except KeyError as e:
            print(f"Error calculating objective: {e}")
            return self.penalty_value
