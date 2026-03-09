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
        win: int,
        baseline_step_pnl: float = 0.0,
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
        win: int | None,
        baseline_step_pnl: float = 0.0,
    ) -> float:
        if is_bankrupt:
            return -1.0

        net_pnl = step_pnl - step_fees
        return net_pnl / equity


class PnLSignalReward(RewardScheme):
    """
    Pure Advantage (Benchmark Tracking)
    Reward = (Agent_Net_PnL - Baseline_Step_PnL) / Equity
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
        baseline_step_pnl: float = 0.0,
    ) -> float:
        if is_bankrupt:
            return -1.0

        agent_net_pnl = step_pnl - step_fees
        advantage = agent_net_pnl - baseline_step_pnl

        return advantage / equity
