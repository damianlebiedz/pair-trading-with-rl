from typing import Annotated, Union, Literal, Any

import pandas as pd
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)
from modules.core.enums import (
    Interval,
    BetaHedge,
    RLModelName,
    ObsSpaceType,
    RLRewards,
    RLPolicyType,
    WandbMode,
)


class Market(BaseModel):
    initial_cash: float = Field(gt=0, description="Starting capital for the backtest.")
    fee_rate: float = Field(
        ge=0, description="Transaction fee rate (e.g., 0.001 for 0.1%)."
    )
    risk_free_rate_annual: float = Field(
        description="Annual risk-free rate used for Sharpe/Sortino ratios."
    )
    interval: Interval = Field(
        description=f"Data timeframe used for the simulation. Options: {[e.value for e in Interval]}"
    )


class Settings(BaseModel):
    vol_window: int = Field(
        gt=0, description="Volatility window size (e.g. 24 = one day in '1h' interval)."
    )
    time_decay_min: float = Field(ge=0, description="Start of Time Decay SL.")
    time_decay_max: float = Field(gt=0, description="End of Time Decay SL.")

    @model_validator(mode="after")
    def validate_time_decay_params(self) -> "Settings":
        if self.time_decay_min > self.time_decay_max:
            raise ValueError("time_decay_params min cannot be greater than max")
        return self


class RL(BaseModel):
    training_subfolder: str = Field(
        description="Name of the folder with training data."
    )
    reward: RLRewards = Field(
        description=f"Type of RL reward. Options: {[e.value for e in RLRewards]}"
    )
    obs_space_type: ObsSpaceType = Field(
        description=f"Type of observation space. Options: {[e.value for e in ObsSpaceType]}"
    )
    passes_per_pair: int = Field(
        description="Number of passes per pair during training."
    )
    seed: int = Field(description="Seed for random number generator.")
    verbose: int = Field(description="Verbosity level in training.")


class PairSelection(BaseModel):
    top_n_factor: int = Field(
        gt=0, description="Factor determining how many top pairs to select."
    )
    start: str = Field(description="Start date for pair selection.")
    end: str = Field(description="End date for pair selection.")

    @model_validator(mode="after")
    def validate_dates(self) -> "PairSelection":
        if pd.to_datetime(self.start) >= pd.to_datetime(self.end):
            raise ValueError("PairSelection: 'start' date must be before 'end' date.")
        return self


class Test(BaseModel):
    beta_start: str = Field(
        description="Lookback window start date for beta calculation."
    )
    start: str = Field(description="Start date for test.")
    end: str = Field(description="End date for test.")

    @model_validator(mode="after")
    def validate_dates(self) -> "Test":
        if pd.to_datetime(self.start) >= pd.to_datetime(self.end):
            raise ValueError("Test: 'start' date must be before 'end' date.")
        if pd.to_datetime(self.beta_start) >= pd.to_datetime(self.start):
            raise ValueError("Test: 'beta_start' date must be before 'start' date.")
        return self


class Performance(BaseModel):
    rl_model_subfolder: str | None = Field(
        default=None,
        description="Name of the folder with RL model, null if running without RL.",
    )
    iterations: int = Field(
        gt=0,
        description="Number of backtest iterations (monthly).",
    )
    beta_hedge: BetaHedge = Field(
        description=f"Hedge ratio mode. Options: {[e.value for e in BetaHedge]}"
    )
    delayed_entry: bool = Field(description="Delayed entry flag.")
    sl_lock: bool = Field(description="SL lock until mean-reversal flag.")
    time_decay_sl: bool = Field(description="Time Decay SL flag.")

    test: Test


class RunBacktest(BaseModel):
    test_start: str = Field(description="Start date for the backtest loop.")
    test_end: str = Field(description="End date for the backtest loop.")

    performance: Performance


class A2CBaseline(BaseModel):
    learning_rate: float = Field(
        description="Step size for the optimizer (learning rate for the A2C policy update)."
    )
    n_steps: int = Field(
        description="Number of forward steps to run for each environment before updating the network (typically small for A2C, e.g., 5)."
    )
    gamma: float = Field(
        description="Discount factor for future rewards (between 0 and 1)."
    )
    ent_coef: float = Field(
        description="Entropy coefficient for the loss calculation. Higher values encourage more exploration."
    )


class A2CAlgo(BaseModel):
    algo_name: Literal[RLModelName.A2C_BASELINE]
    policy_type: Literal[RLPolicyType.MLP_POLICY] = Field(
        default=RLPolicyType.MLP_POLICY,
        description="Fixed policy type for A2C algorithm.",
    )
    params: A2CBaseline


class PolicyKwargs(BaseModel):
    lstm_hidden_size: int = Field(
        description="Size of the hidden state in the LSTM cell."
    )
    n_lstm_layers: int = Field(
        description="Number of stacked LSTM layers (usually 1 is sufficient)."
    )
    shared_lstm: bool = Field(
        description="If true, uses a shared LSTM backbone for both Actor and Critic. If false, creates separate LSTMs."
    )
    enable_critic_lstm: bool = Field(
        description="If true, includes an LSTM layer in the Critic network (only relevant if shared_lstm is false)."
    )


class RecurrentPPO(BaseModel):
    learning_rate: float = Field(
        description="Step size for the optimizer (learning rate for the PPO policy update). Smaller values often yield more stable PPO training."
    )
    n_steps: int = Field(
        description="Number of steps to run for each environment per update (PPO usually requires larger buffers than A2C, e.g., 128 or 256)."
    )
    batch_size: int = Field(
        description="Minibatch size used for each gradient update during the optimization passes."
    )
    n_epochs: int = Field(
        description="Number of optimization epochs (passes over the collected rollout data) when updating the network."
    )
    gamma: float = Field(
        description="Discount factor for future rewards (between 0 and 1)."
    )
    ent_coef: float = Field(
        description="Entropy coefficient for the loss calculation. Higher values encourage more exploration."
    )
    clip_range: float = Field(
        description="Range for clipping the surrogate objective. Prevents overly large policy updates to ensure stability (typically 0.2)."
    )


class PPOAlgo(BaseModel):
    algo_name: Literal[RLModelName.RECURRENT_PPO]
    policy_type: Literal[RLPolicyType.MLP_LSTM_POLICY] = Field(
        default=RLPolicyType.MLP_LSTM_POLICY,
        description="Fixed policy type for Recurrent PPO.",
    )
    params: RecurrentPPO


class RLAlgoDefault(BaseModel):
    rl_algo: RLModelName


class Wandb(BaseModel):
    project: str = Field(description="WandB project name.")
    mode: WandbMode = Field(
        description=f"WandB tracking mode. Options: {[e.value for e in WandbMode]}"
    )


RLAlgoConfig = Annotated[Union[A2CAlgo, PPOAlgo], Field(discriminator="algo_name")]


class Config(BaseModel):
    """
    Main validator.
    """

    model_config = ConfigDict(strict=True, arbitrary_types_allowed=True)

    name: str | None = Field(default=None, description="Name of the run/job for Hydra.")
    defaults: list[str | RLAlgoDefault | dict[str, Any]] | None = Field(
        default=None, description="Hydra defaults list."
    )

    tickers: list[str] = Field(description="List of asset tickers.")
    generate_plots: bool = Field(description="Generate plots if true.")
    z_score_window: int = Field(
        gt=0,
        description="Z-Score lookback window size.",
    )
    entry_threshold: float = Field(description="Z-score threshold to open a position.")
    exit_threshold: float | Literal["-entry_threshold"] = Field(
        description="Z-score threshold to close a position. Can be positive or negative (also equals to -entry_threshold)."
    )
    stop_loss: float | None = Field(
        gt=1,
        description="Stop loss multiplier (e.g., 1.05 for 5% from entry_threshold), null if trade without SL.",
    )

    market: Market = Field(
        description="Market simulation parameters including capital, fees, and timeframe."
    )
    settings: Settings = Field(
        description="General strategy parameters, including volatility and time decay bounds."
    )
    pair_selection: PairSelection = Field(
        description="Configuration for statistical tests and top pair filtering."
    )
    performance: Performance = Field(
        description="Trading logic flags, SL types, and backtest execution parameters."
    )
    rl: RL | None = Field(
        default=None,
        description="Reinforcement Learning environment parameters and training settings.",
    )
    run_backtest: RunBacktest | None = Field(
        default=None,
        description="Execution timeline and explicit settings for the backtest runner.",
    )
    rl_algo: RLAlgoConfig | None = Field(
        default=None,
        description="RL algorithm selection (e.g., A2C, PPO) and its specific hyperparameters.",
    )
    wandb: Wandb | None = Field(
        default=None,
        description="Weights & Biases configuration for experiment tracking and logging.",
    )
