from abc import ABC, abstractmethod
from typing import Literal
import pandas as pd


class ObjectiveScheme(ABC):
    """Abstract base class for objective functions."""

    @abstractmethod
    def calculate(
        self, stats: pd.DataFrame, metric_type: Literal["gross", "net"]
    ) -> float:
        """
        Calculates the objective score based on strategy performance statistics.

        Args:
            stats (pd.DataFrame): DataFrame containing performance statistics (e.g., 'gross', 'net' columns).
            metric_type (Literal["gross", "net"]): The type of metrics to use for calculation.

        Returns:
            float: The calculated objective score (higher is usually better).
        """
        pass


class SortinoWithPenalty(ObjectiveScheme):
    """
    Objective function that maximizes the Annual Sortino Ratio.
    Applies a penalty score if the number of trades is below a specified threshold.

    Args:
        min_trades_per_pair (int): Minimum required number of trades to consider the result valid.
        penalty_value (float, optional): Score to return if constraints are not met. Defaults to -100.0.
    """

    def __init__(self, min_trades_per_pair: int, penalty_value: float = -100.0):
        self.min_trades = min_trades_per_pair
        self.penalty_value = penalty_value

    def calculate(
        self, stats: pd.DataFrame, metric_type: Literal["gross", "net"]
    ) -> float:
        """
        Calculates the Sortino Ratio score with a penalty for insufficient trading activity.

        Args:
            stats (pd.DataFrame): DataFrame containing performance statistics.
                                  Expected to have a column matching `metric_type` (e.g., 'net')
                                  with keys like 'win_count', 'lose_count', 'sortino_ratio_annual'.
            metric_type (Literal["gross", "net"]): The column in `stats` to use ('gross' or 'net').

        Returns:
            float: Annual Sortino Ratio if min_trades condition is met, otherwise `penalty_value`.
        """
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
