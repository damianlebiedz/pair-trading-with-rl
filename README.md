# Pair Trading Framework
### Repository for a research paper currently in progress.
This project implements an advanced pairs trading framework comparing two distinct approaches: **Statistical Arbitrage** and **Reinforcement Learning**.

- [Key Features](#key-features)
- [Installation & Running](#installation--running)
  - [Poetry](#poetry)
  - [Docker](#docker)
- [Project Structure](#project-structure)
- [Data Acquisition](#data-acquisition)
- [Usage & Workflows](#usage--workflows)
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

## Installation & Running

### Poetry

This project uses **Poetry** for dependency management and **Python 3.12**.

Make sure that you have installed:
- Python 3.12: https://www.python.org/downloads/release/python-3120/
- Poetry: https://python-poetry.org/docs/#installation

```bash
# Clone the repository
git clone https://github.com/damianlebiedz/research-paper.git
cd research-paper

# Install dependencies
poetry install
```

### Docker

This project uses Docker to ensure a consistent environment.

Follow a two-step workflow: first, build the **Base Image** containing all dependencies, and then use **Docker Compose** to run specific experiment scripts.

Make sure that you have installed:
- Docker Desktop: https://docs.docker.com/desktop/

#### 1. Build the Base Image

```bash
# Clone the repository
git clone https://github.com/damianlebiedz/research-paper.git
cd research-paper

# Build the Base Image
docker build -t research-paper-base .
```

#### 2. Run Experiments with Docker Compose
Once the image is built, you can run any of the predefined services.
To run a service and see the logs in your terminal:

```bash
docker compose up <service_name>
```
Available Services:
- `train_agent`: Trains the RL model.
- `pair_selection`: Executes the pair selection logic.
- `test_single`: Runs a single backtest on specific parameters.
- `test_multi`: Runs batch testing across multiple models/pairs on specific parameters.
- `opt_and_test_multi`: Performs hyperparameter optimization followed by testing.

#### 3. Development Workflow
- Configuration Changes: The `./config` directory is mapped as a volume. This means you can edit your `.yaml` files, and the changes will be applied instantly when you start a container. No rebuild is required for config changes.

- Persistent Data: Training logs, data, and saved models are stored in `/tensorboard_logs`, `/data`, and `/models` respectively. These folders are synchronized between your machine and the container.

- Overriding Parameters: You can override any Hydra configuration parameter on-the-fly without editing files:

```bash
docker compose run --rm train_agent python runners/train_agent.py rl.obs_space_type=standard
```

#### 4. Multirun
You can use hydra multirun with Docker Compose:
```bash
docker compose run --rm test_multi python runners/test_multi.py -m stop_loss=2,2.5
```

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

## License
Only for research/educational purposes. Commercial use is prohibited. See LICENSE for full terms.