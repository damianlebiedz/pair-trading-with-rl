from abc import ABC, abstractmethod


class RewardScheme(ABC):
    @abstractmethod
    def calculate(
        self,
        step_pnl: float,
        equity: float,
        position: float,
        signal: float,
        step_fees: float,
        is_bankrupt: bool,
        fee_rate: float,
        market_win: int,
    ) -> float:
        pass

    def reset(self):
        pass


class PnLReward(RewardScheme):
    """
    Reward = (step_pnl - step_fees) / equity
    Reward = -1.0 if is_bankrupt
    """

    def calculate(
        self,
        step_pnl: float,
        equity: float,
        position: float | None,
        signal: float | None,
        step_fees: float,
        is_bankrupt: bool,
        fee_rate: float | None,
        market_win: int | None,
    ) -> float:
        if is_bankrupt:
            return -1.0

        net_pnl = step_pnl - step_fees
        return net_pnl / equity


class PnLSignalReward(RewardScheme):
    """
    Reward = (step_pnl - step_fees - penalty) / equity

    penalty = (multiplier * 2 * fee_rate * equity) / market_win
        - multiplier = 2 if position != signal and position != 0 and signal != 0
        - multiplier = 1 if position != signal and position == 0 or signal == 0

    Reward = -1.0 if is_bankrupt
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
        market_win: int,
    ) -> float:
        if is_bankrupt:
            return -1.0

        if position != signal:
            multiplier = (
                2.0 if (signal != 0 and position != 0 and signal != position) else 1.0
            )
            penalty = (multiplier * 2 * fee_rate * equity) / market_win
        else:
            penalty = 0.0

        net_pnl = step_pnl - step_fees - penalty
        return net_pnl / equity
