from typing import Annotated, Union, Literal, Any
import pandas as pd
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
    StringConstraints,
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

DateStr = Annotated[str, StringConstraints(pattern=r"^\d{4}-\d{2}-\d{2}$")]


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
    training_folder: str | None = Field(
        description="Name of the folder with training data, if null take first one."
    )
    reward: RLRewards = Field(
        description=f"Type of RL reward. Options: {[e.value for e in RLRewards]}"
    )
    reward_lambda: float | None = Field(description="Lambda for reward function.")
    fee_multiplier: float | None = Field(
        description="Fee Multiplier for reward function."
    )
    obs_space_type: ObsSpaceType = Field(
        description=f"Type of observation space. Options: {[e.value for e in ObsSpaceType]}"
    )
    passes_per_pair: int = Field(
        description="Number of passes per pair during training."
    )
    seed: int = Field(description="Seed for random number generator.")
    verbose: int = Field(description="Verbosity level in training.")
    freeze_std: bool = Field(
        description="Flag to use fixed std from entry while calculating in-position Z-Score in RL."
    )
    time_decay_stop: bool = Field(
        description="Flag to always close position when time in position is >= Z-Score window."
    )


class PairSelection(BaseModel):
    top_n: int = Field(
        gt=0, description="Factor determining how many top pairs to select."
    )
    start: DateStr = Field(description="Start date for pair selection (YYYY-MM-DD).")
    end: DateStr = Field(description="End date for pair selection (YYYY-MM-DD).")

    @model_validator(mode="after")
    def validate_dates(self) -> "PairSelection":
        if pd.to_datetime(self.start) >= pd.to_datetime(self.end):
            raise ValueError("PairSelection: 'start' date must be before 'end' date.")
        return self


class Performance(BaseModel):
    use_rl: bool = Field(description="Flag to use RL model during backtest.")
    z_score_window: int = Field(
        gt=0,
        description="Z-Score lookback window size.",
    )
    entry_threshold: float = Field(
        gt=0, description="Z-score threshold to open a position."
    )
    exit_threshold: float = Field(description="Z-score threshold to close a position.")
    stop_loss: float | None = Field(
        gt=1,
        description="Stop loss multiplier (e.g., 1.05 for 5% from entry_threshold), null if trade without SL.",
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
    freeze_std: bool = Field(
        description="Flag to use fixed std from entry while calculating in-position Z-Score."
    )
    beta_start: str = Field(
        description="Lookback window start date for beta calculation."
    )
    start: DateStr = Field(description="Start date for test (YYYY-MM-DD).")
    end: DateStr = Field(description="End date for test (YYYY-MM-DD).")

    @model_validator(mode="after")
    def validate_dates(self) -> "Performance":
        if pd.to_datetime(self.start) >= pd.to_datetime(self.end):
            raise ValueError("Test: 'start' date must be before 'end' date.")
        if pd.to_datetime(self.beta_start) >= pd.to_datetime(self.start):
            raise ValueError("Test: 'beta_start' date must be before 'start' date.")
        return self

    @model_validator(mode="after")
    def validate_thresholds(self) -> "Performance":
        if abs(self.exit_threshold) > self.entry_threshold:
            raise ValueError(
                "Test: abs(exit_threshold) cannot be bigger than entry_threshold."
            )
        return self


class DataFetchingPipeline(BaseModel):
    top_n: int = Field(
        gt=0, description="Number of top assets to select based on volume/liquidity."
    )
    start: DateStr = Field(
        description="Start date for evaluating asset liquidity and volume (YYYY-MM-DD)."
    )
    end: DateStr = Field(
        description="End date for evaluating asset liquidity and volume (YYYY-MM-DD)."
    )
    test_end: DateStr = Field(description="End date for test (YYYY-MM-DD).")
    iterations: int = Field(
        gt=0,
        description="Number of iterations (monthly) to fetch historical data.",
    )
    whitelist: list[str] = Field(
        default_factory=list,
        description="List of tickers to forcibly include in the final list.",
    )
    blacklist: list[str] = Field(
        default_factory=list,
        description="List of tickers to forcibly exclude from the final list.",
    )

    @model_validator(mode="after")
    def validate_dates(self) -> "DataFetchingPipeline":
        if pd.to_datetime(self.start) >= pd.to_datetime(self.end):
            raise ValueError(
                "GenerateAssetsList: 'start' date must be before 'end' date."
            )
        return self


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
    policy_kwargs: PolicyKwargs | None = None


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
    clean_single_backtests: bool | None = Field(
        default=False,
        description="Flag to clean the single backtest data ('test' subdirs) during multi-pair/multi-iteration backtesting.",
    )
    generate_plots: bool | None = Field(
        default=False, description="Generate plots if true."
    )
    save_for_training: bool | None = Field(
        default=False,
        description="Flag to auto-save backtest data in data/rl_training for RL training.",
    )
    rl_model_folder: str | None = Field(
        default=None,
        description="Name of the folder with RL model, if null take first one or run without RL.",
    )
    market: Market | None = Field(
        default=None,
        description="Market simulation parameters including capital, fees, and timeframe.",
    )
    settings: Settings | None = Field(
        default=None,
        description="General strategy parameters, including volatility and time decay bounds.",
    )
    pair_selection: PairSelection | None = Field(
        default=None,
        description="Configuration for statistical tests and top pair filtering.",
    )
    performance: Performance | None = Field(
        default=None,
        description="Trading logic flags, SL types, and backtest execution parameters.",
    )
    data_fetching_pipeline: DataFetchingPipeline | None = Field(default=None)
    rl: RL | None = Field(
        default=None,
        description="Reinforcement Learning environment parameters and training settings.",
    )
    rl_algo: RLAlgoConfig | None = Field(
        default=None,
        description="RL algorithm selection (e.g., A2C, PPO) and its specific hyperparameters.",
    )
    wandb: Wandb | None = Field(
        default=None,
        description="Weights & Biases configuration for experiment tracking and logging.",
    )
