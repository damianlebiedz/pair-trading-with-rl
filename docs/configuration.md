# Configuration Documentation (YAML)

Below is an automatically generated list of all configuration parameters supported by the system.

## Root Parameters

- **`tickers`**: List of asset tickers.
- **`generate_plots`**: Generate plots if true.
- **`z_score_window`**: Z-Score lookback window size.
- **`entry_threshold`**: Z-score threshold to open a position.
- **`exit_threshold`**: Z-score threshold to close a position.
- **`stop_loss`**: Stop loss multiplier (e.g., 1.05 for 5% from entry_threshold), null if trade without SL.
- **`market`**: Market simulation parameters including capital, fees, and timeframe.
- **`settings`**: General strategy parameters, including volatility and time decay bounds.
- **`pair_selection`**: Configuration for statistical tests and top pair filtering.
- **`performance`**: Trading logic flags, SL types, and backtest execution parameters.
- **`rl`**: Reinforcement Learning environment parameters and training settings.
- **`run_backtest`**: Execution timeline and explicit settings for the backtest runner.
- **`rl_algo`**: RL algorithm selection (e.g., A2C, PPO) and its specific hyperparameters.
- **`wandb`**: Weights & Biases configuration for experiment tracking and logging.

---

## Configuration Modules

### A2CAlgo
- **`algo_name`**: *No description provided*
- **`policy_type`**: Fixed policy type for A2C algorithm.
- **`params`**: *No description provided*


### A2CBaseline
- **`learning_rate`**: Step size for the optimizer (learning rate for the A2C policy update).
- **`n_steps`**: Number of forward steps to run for each environment before updating the network (typically small for A2C, e.g., 5).
- **`gamma`**: Discount factor for future rewards (between 0 and 1).
- **`ent_coef`**: Entropy coefficient for the loss calculation. Higher values encourage more exploration.


### Market
- **`initial_cash`**: Starting capital for the backtest.
- **`fee_rate`**: Transaction fee rate (e.g., 0.001 for 0.1%).
- **`risk_free_rate_annual`**: Annual risk-free rate used for Sharpe/Sortino ratios.
- **`interval`**: Data timeframe used for the simulation. Options: ['1d', '4h', '1h', '30m', '15m', '5m', '3m', '1m']


### PPOAlgo
- **`algo_name`**: *No description provided*
- **`policy_type`**: Fixed policy type for Recurrent PPO.
- **`params`**: *No description provided*


### PairSelection
- **`top_n_factor`**: Factor determining how many top pairs to select.
- **`start`**: Start date for pair selection.
- **`end`**: End date for pair selection.


### Performance
- **`rl_model_subfolder`**: Name of the folder with RL model, null if running without RL.
- **`iterations`**: Number of backtest iterations (monthly).
- **`beta_hedge`**: Hedge ratio mode. Options: ['no_hedge', 'static', 'rolling']
- **`delayed_entry`**: Delayed entry flag.
- **`sl_lock`**: SL lock until mean-reversal flag.
- **`time_decay_sl`**: Time Decay SL flag.
- **`test`**: *No description provided*


### RL
- **`training_subfolder`**: Name of the folder with training data.
- **`reward`**: Type of RL reward. Options: ['pnl', 'pnl_signal']
- **`obs_space_type`**: Type of observation space. Options: ['minimal', 'standard', 'full']
- **`passes_per_pair`**: Number of passes per pair during training.
- **`seed`**: Seed for random number generator.
- **`verbose`**: Verbosity level in training.


### RLAlgoDefault
- **`rl_algo`**: *No description provided*


### RecurrentPPO
- **`learning_rate`**: Step size for the optimizer (learning rate for the PPO policy update). Smaller values often yield more stable PPO training.
- **`n_steps`**: Number of steps to run for each environment per update (PPO usually requires larger buffers than A2C, e.g., 128 or 256).
- **`batch_size`**: Minibatch size used for each gradient update during the optimization passes.
- **`n_epochs`**: Number of optimization epochs (passes over the collected rollout data) when updating the network.
- **`gamma`**: Discount factor for future rewards (between 0 and 1).
- **`ent_coef`**: Entropy coefficient for the loss calculation. Higher values encourage more exploration.
- **`clip_range`**: Range for clipping the surrogate objective. Prevents overly large policy updates to ensure stability (typically 0.2).


### RunBacktest
- **`test_start`**: Start date for the backtest loop.
- **`test_end`**: End date for the backtest loop.
- **`performance`**: *No description provided*


### Settings
- **`vol_window`**: Volatility window size (e.g. 24 = one day in '1h' interval).
- **`time_decay_min`**: Start of Time Decay SL.
- **`time_decay_max`**: End of Time Decay SL.


### Test
- **`beta_start`**: Lookback window start date for beta calculation.
- **`start`**: Start date for test.
- **`end`**: End date for test.


### Wandb
- **`project`**: WandB project name.
- **`mode`**: WandB tracking mode. Options: ['online', 'offline']

