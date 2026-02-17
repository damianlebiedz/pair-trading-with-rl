# Pairs Trading Framework
### Repository for a research paper currently in progress.
This project implements an advanced pairs trading framework comparing two distinct approaches: **Statistical Arbitrage** and **Reinforcement Learning**.

- [Key Features](#key-features)
- [Installation](#installation)
- [Data Acquisition](#data-acquisition)
- [Usage & Workflows](#usage--workflows)
- [Configuration](#Configuration)
- [Project Structure](#project-structure)
- [License](#license)

## Key Features

### 1. Statistical & Optimization Pipeline
- **Pair Selection:** Automated search for cointegrated pairs with **Hurst Exponent** filter.
- **Optimization:** Hyperparameter tuning using **Random Search** to find optimal entry/exit thresholds and window sizes.
- **Walk-Forward Analysis:** Robust backtesting engine with rolling windows (Selection → Optimization → Testing).

### 2. Reinforcement Learning Pipeline
- **Custom Environment:** OpenAI Gym (`gymnasium`) compatible `PairsTradingEnv`.
- **Algorithms:** Support for A2C (Stable Baselines 3) and Recurrent PPO (SB3 Contrib).
- **Reward Engineering:** Implements various reward functions: PnL, Risk-Adjusted, Volatility Penalty, and Differential Sharpe Ratio.
- **Monitoring:** Integrated with **Weights & Biases (WandB)** for experiment tracking.

---

## Installation

This project uses **Poetry** for dependency management and **Python 3.12**.

```bash
# Clone the repository
git clone https://github.com/damianlebiedz/research-paper.git
cd research-paper

# Install dependencies
poetry install
```

## Data Acquisition
Before running any pipelines, you need to fetch historical market data. A helper script is provided to download OHLCV data directly from Binance.

```bash
poetry run python helpers/data_fetching.py
```
Data is saved to `data/{TICKER}/{filename}.csv`.

## Usage & Workflows
The project entry points are located in the `runners/` directory. Configuration is managed via Hydra.

1. Statistical Optimization & Testing

Runs the full pipeline: Pair Selection → Hyperparameter Optimization → Out-of-Sample Testing.

```bash
poetry run python runners/opt_and_test_multi.py
```
Config location: `config/opt_and_test_multi.yaml`

2. Reinforcement Learning Agent Training

Trains an RL agent on a specific dataset.

```bash
# Make sure to configure your WandB API key in .env
poetry run python runners/train_agent.py
```
Config location: `config/train_agent.yaml`

## Configuration
The project currently uses standard **Hydra** YAML configuration files located in the `config/` directory.

- `base.yaml`: General global settings.
- `rl_algo/`: Specific configurations for A2C/PPO.
- `opt_and_test_multi.yaml`: Settings for the statistical pipeline.

## Project Structure
```
.
├── config/                 # Hydra configuration files
├── data/                   # Market data (downloaded via helpers)
├── helpers/                # Scripts for data fetching and reporting
├── modules/
│   ├── core/               # Execution logic, indicators, stat tests
│   ├── data_services/      # Data loading and merging utils
│   ├── learning/           # RL environments, agents, and rewards
│   └── performance/        # Strategy logic, optimization objectives
├── runners/                # Main entry points (scripts)
```

## License
Only for research/educational purposes. Commercial use is prohibited. See LICENSE for full terms.