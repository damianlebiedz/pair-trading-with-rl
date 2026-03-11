import warnings
from abc import ABC

from modules.core.enums import RLRewards


class RewardScheme(ABC):
    """
    Abstract base class for Reinforcement Learning reward schemes.
    Automatically validates if the subclass exists in the RLRewards Enum.
    """

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        allowed_reward_names = [enum_item.value for enum_item in RLRewards]

        if cls.__name__ not in allowed_reward_names:
            warnings.warn(
                f"\n\n[WARNING]: Class '{cls.__name__}' is not defined in the 'RLRewards' Enum!\n"
                f"Please add it to 'enums.py' to enable YAML auto-completion.\n",
                UserWarning,
                stacklevel=2,
            )

    def reset(self):
        pass

    def calculate(self, *args, **kwargs) -> float:
        raise NotImplementedError


class PnLReward(RewardScheme):
    """
    Standard Net Profit and Loss (PnL) reward function.
    Calculates the step return normalized by the current equity.

    Formula:
        R_t = (PnL_t - Fees_t) / Equity_t

    Terminal State:
        R_t = -1.0 if bankrupt.
    """

    def calculate(
        self,
        step_pnl: float,
        equity: float,
        position: float,
        signal: float,
        step_fees: float,
        is_bankrupt: bool,
        fee_rate: float,
        win: int,
    ) -> float:
        if is_bankrupt:
            return -1.0

        net_pnl = step_pnl - step_fees
        return float(net_pnl / equity)


class AsymmetricReward(RewardScheme):
    """
    Reward function with Asymmetric Loss Aversion (based on Prospect Theory).
    Penalizes negative returns strictly harder than it rewards positive ones
    to prevent risky trades and false breakout traps.

    Formula:
        r_t = (PnL_t - Fees_t) / Equity_t
        R_t = r_t if r_t >= 0 else r_t * lambda
        (where lambda = 2.0)
    """

    def reset(self):
        pass

    def calculate(
        self,
        step_pnl: float,
        equity: float,
        position: float,
        signal: float,
        step_fees: float,
        is_bankrupt: bool,
        fee_rate: float,
        win: int,
    ) -> float:
        if is_bankrupt:
            return -1.0

        reward = (step_pnl - step_fees) / equity
        if reward < 0:
            reward *= 2.0

        return float(reward)


class CompositeReward(RewardScheme):
    """
    Composite reward function applying both Asymmetric Loss Aversion
    and an Action Churn Penalty to enforce policy stability.
    Discourages High-Frequency Trading (HFT) and reduces slippage impact.

    Formula:
        r_t = (PnL_t - Fees_t) / Equity_t
        R_t = (r_t if r_t >= 0 else r_t * 2.0) - (c * I[a_t != a_{t-1}])
        (where c is a fraction of the standard fee rate, e.g., 0.2 * fee_rate)
    """

    def __init__(self):
        super().__init__()
        self.prev_position = 0.0

    def reset(self):
        self.prev_position = 0.0

    def calculate(
        self,
        step_pnl: float,
        equity: float,
        position: float,
        signal: float,
        step_fees: float,
        is_bankrupt: bool,
        fee_rate: float,
        win: int,
    ) -> float:
        if is_bankrupt:
            return -1.0

        reward = (step_pnl - step_fees) / equity

        # 1. Asymmetric Loss Aversion
        if reward < 0:
            reward *= 2.0

        if position != self.prev_position:
            reward -= fee_rate * 0.2

        self.prev_position = position
        return float(reward)
