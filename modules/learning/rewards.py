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


class AsymmetricReward(RewardScheme):
    """
    Universal reward function with configurable Asymmetric Loss Aversion.
    Based on Prospect Theory (Kahneman & Tversky).

    If reward_lambda = 1.0, it acts as a standard Risk-Neutral PnLReward.
    If reward_lambda > 1.0, it penalizes negative returns strictly harder
    than it rewards positive ones.

    Formula:
        r_t = (PnL_t - Fees_t) / Equity_t
        R_t = r_t if r_t >= 0 else r_t * reward_lambda
    """

    def __init__(self, reward_lambda: float = 1.0):
        super().__init__()
        self.reward_lambda = reward_lambda

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
            reward *= self.reward_lambda

        return float(reward)
