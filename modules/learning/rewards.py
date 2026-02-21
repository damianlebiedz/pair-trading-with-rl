from abc import ABC, abstractmethod
import numpy as np


class RewardScheme(ABC):
    @abstractmethod
    def calculate(
        self,
        step_pnl: float,
        equity: float,
        position: float,
        step_fees: float,
        info: dict,
    ) -> float:
        pass

    def reset(self):
        pass


class PnLReward(RewardScheme):
    def calculate(
        self,
        step_pnl: float,
        equity: float,
        position: float,
        step_fees: float,
        info: dict,
    ) -> float:
        return step_pnl


class RiskAdjustedReward(RewardScheme):
    def __init__(self, risk_penalty: float = 0.1):
        self.risk_penalty = risk_penalty

    def calculate(
        self,
        step_pnl: float,
        equity: float,
        position: float,
        step_fees: float,
        info: dict,
    ) -> float:
        drawdown_pct = info.get("drawdown_pct", 0.0)
        return step_pnl - (drawdown_pct * self.risk_penalty)


class DifferentialSharpeReward(RewardScheme):
    """Differential Sharpe Ratio (DSR) by Moody, Saffell (2001)."""

    def __init__(self, decay_rate: float = 0.01):
        """
        Args:
            decay_rate (eta): e.g. 0.01 ~ window 100.
        """
        self.decay_rate = decay_rate
        # A_t: returns EMA
        self.A_t = 0.0
        # B_t: returns^2 EMA
        self.B_t = 0.0
        self.initialized = False

    def reset(self):
        self.A_t = 0.0
        self.B_t = 0.0
        self.initialized = False

    def calculate(
        self,
        step_pnl: float,
        equity: float,
        position: float,
        step_fees: float,
        info: dict,
    ) -> float:
        if equity <= 1e-6:
            return -1.0  # bankrupt penalty

        r_t = step_pnl / equity

        if not self.initialized:
            self.A_t = r_t
            self.B_t = r_t**2
            self.initialized = True
            return 0.0

        prev_A = self.A_t
        prev_B = self.B_t

        # Update EMA
        # A_t = A_{t-1} + eta * (r_t - A_{t-1})
        self.A_t = prev_A + self.decay_rate * (r_t - prev_A)
        # B_t = B_{t-1} + eta * (r_t^2 - B_{t-1})
        self.B_t = prev_B + self.decay_rate * (r_t**2 - prev_B)

        variance = prev_B - prev_A**2

        if variance < 1e-9:
            return 0.0

        std_dev = np.sqrt(variance)

        # Differential Sharpe Ratio:
        # D_t = [ B_{t-1} * (r_t - A_{t-1}) - 0.5 * A_{t-1} * (r_t^2 - B_{t-1}) ] / (Variance ^ 1.5)
        term1 = prev_B * (r_t - prev_A)
        term2 = 0.5 * prev_A * (r_t**2 - prev_B)
        denominator = variance * std_dev  # variance ^ 1.5

        dsr = (term1 - term2) / denominator

        # Optionally: return np.tanh(dsr)
        return dsr
