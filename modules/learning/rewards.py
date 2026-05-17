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


class StepPnLReward(RewardScheme):
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

    def __init__(self, reward_lambda: float = 1.0, **_):
        super().__init__()
        self.reward_lambda = reward_lambda

    def reset(self):
        pass

    def calculate(
        self,
        step_pnl: float,
        equity: float,
        step_fees: float,
        is_bankrupt: bool,
        **_,
    ) -> float:
        if is_bankrupt:
            return -1.0

        reward = (step_pnl - step_fees) / equity

        if reward < 0:
            reward *= self.reward_lambda

        return float(reward)


class TradePnLReward(RewardScheme):
    """
    Sparse, trade-based reward with configurable Asymmetric Loss Aversion.

    Unlike StepPnLReward, the agent receives feedback only when a position is
    closed (trade_ended), ignoring intra-trade drawdowns and temporary noise.
    Based on Prospect Theory (Kahneman & Tversky) and risk-sensitive RL
    (Mihatsch & Neuneier, 2002).

    If reward_lambda = 1.0, finalized gains and losses are weighted symmetrically.
    If reward_lambda > 1.0, losing trades are penalized more heavily than
    winning trades are rewarded.

    Formula:
        r_trade = PnL_trade / Equity_t
        R_t = r_trade           if trade_ended and r_trade >= 0
            = r_trade * lambda  if trade_ended and r_trade < 0
            = 0                 otherwise
    """

    def __init__(self, reward_lambda: float = 1.0, **_):
        super().__init__()
        self.reward_lambda = reward_lambda

    def calculate(
        self,
        equity: float,
        is_bankrupt: bool,
        trade_ended: bool,
        trade_pnl: float,
        **_,
    ) -> float:
        if is_bankrupt:
            return -1.0

        if not trade_ended:
            return 0.0

        reward = trade_pnl / equity

        if reward < 0:
            reward *= self.reward_lambda

        return float(reward)


class HybridActionReward(RewardScheme):
    """
    Hybrid reward combining sparse TradePnL with dense signal-action shaping.

    Realized trade PnL uses the same asymmetric scaling as TradePnLReward.
    On entry from flat (prev_position == 0), an immediate component Phi_t
    reinforces alignment with the baseline mean-reversion signal (S_t) relative
    to the round-trip fee 2 * fee_rate and tuning multiplier fee_multiplier (m).

    This scheme tests guided exploration: omission and contrarian penalties
    embed the baseline heuristic into the reward landscape rather than training
    a fully autonomous policy (Yang et al., 2024).

    Default fee_multiplier = 0.2 yields an action bonus of 40% of the
    round-trip transaction cost (Phi_t = +2 * fee_rate * m when A_t = S_t).

    Entry shaping (only when prev_position == 0 and signal != 0):
        Phi_t = +2 * fee_rate * m       if curr_position == signal  (Action Bonus)
              = -2 * fee_rate * m       if curr_position == 0        (Omission Penalty)
              = -4 * fee_rate * m       otherwise                    (Contrarian Penalty)

    On trade close:
        R_trade = trade_pnl / equity, scaled by reward_lambda if negative.

    Formula:
        R_t = R_trade + Phi_t
    """

    def __init__(self, reward_lambda: float = 1.0, fee_multiplier: float = 0.2, **_):
        super().__init__()
        self.reward_lambda = reward_lambda
        self.fee_multiplier = fee_multiplier

    def calculate(
        self,
        equity,
        prev_position,
        curr_position,
        signal,
        is_bankrupt,
        fee_rate,
        trade_ended=False,
        trade_pnl=0.0,
        **_,
    ):
        if is_bankrupt:
            return -1.0

        reward = 0.0
        total_entry_fee = fee_rate * 2.0
        dynamic_action_bonus = total_entry_fee * self.fee_multiplier

        if prev_position == 0.0:
            if signal != 0.0:
                if curr_position == signal:
                    reward += dynamic_action_bonus
                elif curr_position == 0.0:
                    reward -= dynamic_action_bonus
                else:
                    reward -= dynamic_action_bonus * 2

        if trade_ended:
            base_reward = trade_pnl / equity
            reward += (
                base_reward if base_reward > 0 else base_reward * self.reward_lambda
            )

        return float(reward)
