"""Full coverage tests for modules.learning.rewards."""

from __future__ import annotations

import warnings

import pytest

from modules.learning.rewards import (
    HybridActionReward,
    RewardScheme,
    StepPnLReward,
    TradePnLReward,
)


class TestRewardScheme:
    def test_init_subclass_warns_for_unregistered_name(self) -> None:
        with pytest.warns(UserWarning, match="RLRewards"):

            class _UnregisteredReward(RewardScheme):
                pass

    def test_reset_is_noop(self) -> None:
        scheme = StepPnLReward()
        scheme.reset()

    def test_calculate_not_implemented_on_base_subclass(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)

            class _BareReward(RewardScheme):
                pass

        with pytest.raises(NotImplementedError):
            _BareReward().calculate()


class TestStepPnLReward:
    def test_bankrupt(self) -> None:
        reward = StepPnLReward()
        assert reward.calculate(0, 1000, 0, is_bankrupt=True) == -1.0

    def test_positive_reward(self) -> None:
        reward = StepPnLReward(reward_lambda=2.0)
        assert reward.calculate(100, 1000, 10, is_bankrupt=False) == pytest.approx(0.09)

    def test_negative_reward_scaled_by_lambda(self) -> None:
        reward = StepPnLReward(reward_lambda=2.0)
        result = reward.calculate(-100, 1000, 10, is_bankrupt=False)
        assert result == pytest.approx(-0.22)

    def test_reset(self) -> None:
        StepPnLReward().reset()


class TestTradePnLReward:
    def test_bankrupt(self) -> None:
        reward = TradePnLReward()
        assert (
            reward.calculate(1000, is_bankrupt=True, trade_ended=True, trade_pnl=50)
            == -1.0
        )

    def test_no_trade_no_reward(self) -> None:
        reward = TradePnLReward()
        assert (
            reward.calculate(1000, is_bankrupt=False, trade_ended=False, trade_pnl=50)
            == 0.0
        )

    def test_positive_trade_reward(self) -> None:
        reward = TradePnLReward()
        assert reward.calculate(
            1000, is_bankrupt=False, trade_ended=True, trade_pnl=100
        ) == pytest.approx(0.1)

    def test_negative_trade_reward_scaled(self) -> None:
        reward = TradePnLReward(reward_lambda=3.0)
        assert reward.calculate(
            1000, is_bankrupt=False, trade_ended=True, trade_pnl=-50
        ) == pytest.approx(-0.15)


class TestHybridActionReward:
    def test_bankrupt(self) -> None:
        reward = HybridActionReward()
        assert (
            reward.calculate(
                equity=1000,
                prev_position=0,
                curr_position=1,
                signal=1,
                is_bankrupt=True,
                fee_rate=0.001,
            )
            == -1.0
        )

    def test_no_entry_shaping_when_already_in_position(self) -> None:
        reward = HybridActionReward(fee_multiplier=0.2)
        assert (
            reward.calculate(
                equity=1000,
                prev_position=1.0,
                curr_position=1.0,
                signal=1.0,
                is_bankrupt=False,
                fee_rate=0.001,
            )
            == 0.0
        )

    def test_no_entry_shaping_when_signal_zero(self) -> None:
        reward = HybridActionReward()
        assert (
            reward.calculate(
                equity=1000,
                prev_position=0.0,
                curr_position=0.0,
                signal=0.0,
                is_bankrupt=False,
                fee_rate=0.001,
            )
            == 0.0
        )

    def test_action_bonus(self) -> None:
        reward = HybridActionReward(fee_multiplier=0.2)
        assert reward.calculate(
            equity=1000,
            prev_position=0.0,
            curr_position=1.0,
            signal=1.0,
            is_bankrupt=False,
            fee_rate=0.001,
        ) == pytest.approx(0.0004)

    def test_omission_penalty(self) -> None:
        reward = HybridActionReward(fee_multiplier=0.2)
        assert reward.calculate(
            equity=1000,
            prev_position=0.0,
            curr_position=0.0,
            signal=1.0,
            is_bankrupt=False,
            fee_rate=0.001,
        ) == pytest.approx(-0.0004)

    def test_contrarian_penalty(self) -> None:
        reward = HybridActionReward(fee_multiplier=0.2)
        assert reward.calculate(
            equity=1000,
            prev_position=0.0,
            curr_position=-1.0,
            signal=1.0,
            is_bankrupt=False,
            fee_rate=0.001,
        ) == pytest.approx(-0.0008)

    def test_trade_close_positive(self) -> None:
        reward = HybridActionReward()
        assert reward.calculate(
            equity=1000,
            prev_position=1.0,
            curr_position=0.0,
            signal=0.0,
            is_bankrupt=False,
            fee_rate=0.001,
            trade_ended=True,
            trade_pnl=100,
        ) == pytest.approx(0.1)

    def test_trade_close_negative_scaled(self) -> None:
        reward = HybridActionReward(reward_lambda=2.0)
        assert reward.calculate(
            equity=1000,
            prev_position=1.0,
            curr_position=0.0,
            signal=0.0,
            is_bankrupt=False,
            fee_rate=0.001,
            trade_ended=True,
            trade_pnl=-50,
        ) == pytest.approx(-0.1)
