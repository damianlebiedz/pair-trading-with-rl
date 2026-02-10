from abc import ABC, abstractmethod


class RewardScheme(ABC):
    @abstractmethod
    def calculate(self, step_pnl: float, equity: float, position: float, step_fees: float, info: dict) -> float:
        pass

    def reset(self):
        pass


class PnLReward(RewardScheme):
    def calculate(self, step_pnl: float, equity: float, position: float, step_fees: float, info: dict) -> float:
        return step_pnl


class RiskAdjustedReward(RewardScheme):
    def __init__(self, risk_penalty: float = 0.1):
        self.risk_penalty = risk_penalty

    def calculate(self, step_pnl: float, equity: float, position: float, step_fees: float, info: dict) -> float:
        drawdown_pct = info.get('drawdown_pct', 0.0)
        return step_pnl - (drawdown_pct * self.risk_penalty)
