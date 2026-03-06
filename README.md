# Pair Trading Framework
### Repository for a research paper currently in progress.
This project implements an advanced pairs trading framework comparing two distinct approaches: **Statistical Arbitrage** and **Reinforcement Learning**.

- [Key Features](#key-features)
- [Installation & Setup](#installation--setup)
  - [Docker](#docker)
  - [Poetry](#poetry)
- [Reproducing Results (Paper Evaluation)](#reproducing-results-paper-evaluation)
- [Project Structure](#project-structure)
- [Configuration](#Configuration)
- [License](#license)

## Key Features

### 1. Statistical & Optimization Pipeline
- **Pair Selection:** Automated search for cointegrated pairs with Hurst Exponent filter.
- **Optimization:** Hyperparameter tuning using **Random Search** to find optimal entry/exit thresholds and window sizes.
- **Walk-Forward Analysis:** Robust backtesting engine with rolling windows (Selection → Optimization → Testing).

### 2. Reinforcement Learning Pipeline
- **Custom Environment:** OpenAI Gym (`gymnasium`) compatible `PairsTradingEnv`.
- **Algorithms:** Support for A2C (Stable Baselines 3) and Recurrent PPO (SB3 Contrib).
- **Reward Engineering:** Implements various reward functions: PnL, Risk-Adjusted, and Differential Sharpe Ratio.
- **Monitoring:** Integrated with **Weights & Biases (WandB)** for experiment tracking.

---

## Installation & Setup

First, clone the repository and configure your environment variables. You must set up your Weights & Biases (WandB) API key, which is required for tracking the Reinforcement Learning training process.

```bash
# Clone the repository
git clone [https://github.com/damianlebiedz/research-paper.git](https://github.com/damianlebiedz/research-paper.git)
cd research-paper

# Set up environment variables
cp .env.example .env
# Open the .env file and add your actual API key: WANDB_API_KEY=your_key_here
```

Once the .env file is ready, you can run this project using either Docker (recommended for isolated, reproducible environments) or locally via Poetry.

### Docker

This project uses Docker Compose with a base image to ensure 100% environment consistency.

```bash
# Build the Base Image
docker compose build
```

### Poetry

Ensure you have Python 3.12 and Poetry installed.

```bash
# Install dependencies
poetry install
```

## Reproducing Results (Paper Evaluation)
To guarantee academic reproducibility, this project uses a Makefile pipeline.

Why? Relying on live external APIs (like Binance) for historical data can lead to inconsistencies due to changing limits or delisted assets. Furthermore, Reinforcement Learning training can introduce hardware-dependent variance.

To solve this, we provide two separate execution tracks: the **Fast Track** (using frozen artifacts) and the **Full Track** (running everything from scratch).

Note: You must specify the environment by appending `RUN_MODE=docker` (recommended for reviewers) or `RUN_MODE=poetry` to your make commands.

### 1. Fast Track (Recommended for Reviewers)
This track evaluates the pre-trained RL agent on a frozen, version-controlled dataset hosted on Zenodo (DOI: 10.5281/zenodo.XXXXXXX). It bypasses the Binance API and the lengthy RL training process, guaranteeing identical results to those published in the paper.

```bash
make fast_track RUN_MODE=docker
```
What it does:
- `download_artifacts`: Downloads and extracts the exact historical/.parquet dataset and the pre-trained .zip RL model from Zenodo.
- `backtest`: Runs the baseline Statistical Arbitrage backtest.
- `backtest_rl`: Runs the backtest using the pre-trained Reinforcement Learning agent.

### 2. Full Track (End-to-End Reproduction)
This track executes the entire pipeline from scratch. It is intended for researchers who want to fetch the latest data or retrain the agent entirely.

```bash
make full_track RUN_MODE=docker
```
What it does:
- `fetch`: Dynamically generates the asset universe and fetches raw OHLCV data directly from the Binance API.
- `backtest`: Runs the baseline Statistical Arbitrage backtest.
- `train`: Trains the Reinforcement Learning agent from scratch (can take several hours).
- `backtest_rl`: Evaluates your newly trained agent.

(Optional) You can also run individual stages, for example: make train `RUN_MODE=docker`.

## Project Structure
```
.
├── config/                 # Hydra configuration files (.yaml & .json schemas)
├── data/                   # Market data (downloaded via helpers or Zenodo)
├── helpers/                # Scripts for data fetching and schema generationreporting
├── modules/
│   ├── core/               # Execution logic, indicators, stat tests
│   ├── data_services/      # Data loading and merging utils
│   ├── learning/           # RL environments, agents, and rewards
│   └── performance/        # Strategy logic, optimization objectives
├── runners/                # Main entry points (backtesting, training)
└── results/                # Output metrics, plots, and saved models
```

## Configuration
The project currently uses standard **Hydra** YAML configuration files located in the `config/` directory.

- `base.yaml`: General global settings.
- `run_backtest.yaml`: Settings for the backtesting engine.
- `rl_algo/`: Specific configurations for A2C and Recurrent PPO.
- `opt_and_test_multi.yaml`: Settings for the statistical pipeline.

You can dynamically override parameters directly from the CLI without editing files. For example, to run the RL backtest manually with a custom configuration:

```bash
docker compose run --rm run_backtest python runners/run_backtest.py use_rl=True
```

## License
Only for research/educational purposes. Commercial use is prohibited. See LICENSE for full terms.