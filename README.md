# Pair Trading Research Framework

Repository for the paper **"Dynamic Multi-Pair Trading Strategy in Cryptocurrency Markets with Deep Reinforcement Learning"** (Lebiedź and Ślepaczuk, 2026). It implements a **statistical arbitrage** backtesting framework and **reinforcement learning** training and evaluation pipelines used in that study.

| |                                                      |
|---|------------------------------------------------------|
| **Paper** | *Coming soon - DOI/link will be added after publication* |
| **Preprint** | `https://arxiv.org/abs/XXXXXXXX` *(placeholder)*     |
| **Frozen artifacts (Zenodo)** | `https://zenodo.org/records/20355140` |
| **Zenodo DOI** | `10.5281/zenodo.20355140`           |

### Fast path

If you only want to **inspect** the paper outputs, pick one of two paths:

**Option A - Manual download (no setup required, any OS).** From the [Zenodo record](https://zenodo.org/records/20355140), grab the two archives and extract them into the repo root:

- [`data.zip`](https://zenodo.org/records/20355140/files/data.zip) → creates `./data/`
- [`results.zip`](https://zenodo.org/records/20355140/files/results.zip) → creates `./results/`

Each archive already contains the top-level folder name, so extracting in the project root puts files in the right place. Use any built-in unzipper (Windows Explorer, macOS Finder, Linux file manager).

**Option B - Scripted (requires Python or Docker).** After completing [Quick start](#quick-start) (`poetry install` or `docker compose build`):

```bash
poetry run python helpers/download_artifacts.py
# or: docker compose run --rm download_artifacts
```

Both options give you the same archives:

| Archive | What you get                                                                                                                                                          |
|---------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **`data/`** | `pair_selection/`, `rl_training/`, and `rl_models/`. **Does not include `historical/`** - see below.                                                                  |
| **`results/`** | Parquet outputs for every experiment cited in the paper (per-pair returns, trades, stats), plus `.hydra/config.yaml` and `.hydra/overrides.yaml` for reproducibility. |

> **`data/historical/` is not on Zenodo.** OHLCV from [Binance Data Vision](https://data.binance.vision/) is excluded. To **re-run** backtests you must fetch it locally (this step *does* require Python or Docker):

```bash
poetry run python helpers/data_fetching_pipeline.py
# or: docker compose run --rm data_fetching_pipeline
```

**Re-run backtests:** download artifacts (Option A or B) → `data_fetching_pipeline` → `run_backtest`. Details: [Quick start](#quick-start) and [Frozen artifacts (Zenodo)](#frozen-artifacts-zenodo).

___

**Stack:** Python 3.12 · Poetry · Hydra · Pydantic · pandas · statsmodels · scikit-optimize · Gymnasium · Stable-Baselines3 · SB3-Contrib (Recurrent PPO) · Weights & Biases · joblib (parallel sweeps) · Docker

- [Quick start](#quick-start)
- [Typical workflows](#typical-workflows)
- [Project structure](#project-structure)
- [Data directory](#data-directory)
- [Configuration (Hydra + Pydantic)](#configuration-hydra--pydantic)
- [Helpers](#helpers)
- [Runners](#runners)
- [Results layout](#results-layout)
- [Multirun & hyperparameter grids](#multirun--hyperparameter-grids)
- [Documentation](#documentation)
- [Frozen artifacts (Zenodo)](#frozen-artifacts-zenodo)
- [Tests & CI](#tests--ci)
- [License](#license)

---

## Quick start

**Prerequisites:** Python 3.12, [Poetry](https://python-poetry.org/), and optionally [Docker](https://docs.docker.com/). For RL training you also need a free [Weights & Biases](https://wandb.ai) account and API key.

```bash
git clone https://github.com/damianlebiedz/research-paper.git
cd research-paper
cp .env.example .env   # set WANDB_API_KEY (required for RL training monitoring)

poetry install
poetry run python helpers/update_config.py          # refresh schemas + docs/configuration.md
poetry run python helpers/data_fetching_pipeline.py # fetch data + list_of_assets (long-running)
poetry run python runners/run_backtest.py
```

What each step does:

1. **`poetry install`** - installs all Python dependencies into a local virtualenv.
2. **`update_config.py`** - syncs JSON schemas and `docs/configuration.md` with the Pydantic models (run once after clone, and again if you change config fields in code).
3. **`data_fetching_pipeline.py`** - downloads historical futures data from Binance Data Vision into `data/historical/`, builds monthly asset universes, and writes `config/schemas/list_of_assets.json`. This step is **always required before re-running backtests** - `data/historical/` is not on Zenodo (see [Fast path](#fast-path)). Other `data/` folders can be fetched from Zenodo instead of recomputing.
4. **`run_backtest.py`** - runs the main walk-forward backtest with defaults from `config/run_backtest.yaml`. Outputs go to `results/run_backtest_<timestamp>/`.

To change parameters without editing YAML, append Hydra overrides, for example:

```bash
poetry run python runners/run_backtest.py performance.entry_threshold=3.0 pair_selection.top_n=20
```

### Docker

Docker gives the same environment on every machine. Build once, then **prepend** `docker compose run --rm <service>` instead of `poetry run python ...`. Hydra CLI overrides work the same way.

```bash
docker compose build
docker compose run --rm update_config
docker compose run --rm data_fetching_pipeline
docker compose run --rm run_pair_selection
docker compose run --rm run_backtest
docker compose run --rm train_agent
```

Override the default service command when needed (e.g. multirun):

```bash
docker compose run --rm run_backtest python runners/run_backtest.py -m hydra/launcher=joblib hydra.launcher.n_jobs=-1 performance.entry_threshold=2.0,2.5,3.0
```

Volumes `config/`, `data/`, and `results/` are mounted from the host, so outputs persist after the container exits.

---

## Typical workflows

Pick the path that matches your goal:

| Goal | What to do                                                                                                                                                                                             |
|------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Inspect paper results only** | `download_artifacts` → browse `results/` (no code to run). |
| **Re-run backtests with paper inputs** | `download_artifacts` → `data_fetching_pipeline` (historical OHLCV) → `run_backtest` (uses Zenodo `pair_selection/` + `rl_models/`). |
| **Full pipeline from scratch** | `data_fetching_pipeline` → statistical `run_backtest` (optionally with multirun grids) → `run_backtest` with `save_for_training=true` → `train_agent` → `run_backtest` with `performance.use_rl=true`. |
| **Only statistical arbitrage** | Skip RL: `performance.use_rl=false` everywhere; no WandB key required for backtests only.                                                                                                              |
| **Hyperparameter search** | Use Hydra multirun + joblib launcher - see [Multirun & hyperparameter grids](#multirun--hyperparameter-grids).                                                                                         |
| **Tables/plots for the paper** | Run core runners first, then optional scripts in `helpers/analysis_scripts/` (see [Helpers](#helpers)).                                                                                                |

---

## Project structure

High-level map of the repository:

```
.
├── config/                      # Hydra YAML + JSON schemas (IDE autocomplete / validation)
│   ├── base.yaml                # shared defaults (market, wandb, hydra dirs)
│   ├── run_backtest.yaml        # statistical / RL backtest runner
│   ├── train_agent.yaml         # RL training runner
│   ├── rl_algo/                 # A2C / Recurrent PPO presets
│   ├── helpers/                 # helper-specific YAML (data pipeline)
│   └── schemas/                 # generated JSON schemas + list_of_assets.json
├── data/                        # all on-disk inputs between runs (see below)
│   ├── historical/              # raw OHLCV from data_fetching_pipeline
│   ├── pair_selection/          # cached cointegration / pair ranks per month
│   ├── rl_training/             # per-pair backtest exports for RL (save_for_training)
│   └── rl_models/               # trained SB3 checkpoints (.zip + VecNormalize)
├── docs/                        # generated & reference docs (see below)
├── helpers/
│   ├── update_config.py         # schema + configuration.md generator
│   ├── data_fetching_pipeline.py
│   └── analysis_scripts/        # optional post-hoc analysis (paper-specific)
├── modules/                     # core library (strategy, RL, data, stats)
├── runners/
│   ├── run_backtest.py
│   ├── train_agent.py
│   └── run_pair_selection.py
├── results/                     # run outputs (timestamped; not committed)
└── tests/
```

- **`modules/`** - reusable logic: indicators, pair selection, strategy execution, RL environment, data I/O. Runners are thin entry points that wire config to these modules.
- **`runners/`** - scripts you actually execute (`run_backtest`, `train_agent`, …).
- **`config/`** - all experiment settings as YAML; edit here or override from the command line.
- **`helpers/`** - standalone scripts for data download, schema generation, and (optionally) post-processing of `results/`.
- **`data/`** - persistent inputs used across runs (market data, pair lists, RL datasets, models). Not the same as `results/` (which stores outputs of a single experiment run). See [Data directory](#data-directory) below.
- **`results/`** - created at runtime; each run gets its own timestamped folder plus a `.hydra/` snapshot of the exact config used.

### Data directory

Everything under `data/` is **input** to the pipelines (or an intermediate cache). Nothing here is a final paper table - those live under `results/`. Typical layout after a full setup:

```
data/
├── historical/
│   ├── .cache/                          # temporary monthly downloads (removed after merge)
│   ├── BTCUSDT_1h_2023-11-01-2024-02-01.parquet
│   ├── ETHUSDT_1h_2023-11-01-2024-02-01.parquet
│   └── ...                              # one file per symbol × interval × date span
├── pair_selection/
│   ├── .hydra/                          # config snapshot from run_pair_selection
│   ├── 2023-11/
│   │   └── pair_selection_2023-11-01_2024-01-01.parquet
│   ├── 2023-12/
│   │   └── pair_selection_2023-12-01_2024-02-01.parquet
│   └── ...
├── rl_training/
│   └── run_backtest_2024-03-01_12-00-00_abc123/   # folder name = backtest run id
│       ├── returns_BTCUSDT_ETHUSDT_2024-01-01_2024-02-01.parquet
│       └── ...                          # one file per pair (test-window trajectories)
└── rl_models/
    ├── recurrent_ppo_autonomous_StepPnLReward_<wandb_run_id>_seed42.zip
    ├── recurrent_ppo_autonomous_StepPnLReward_<wandb_run_id>_seed42_normalize.pkl
    └── ...
```

| Path | Created by | Used by | Contents                                                                                                                                                                                                                                                                          |
|------|------------|---------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **`historical/`** | `helpers/data_fetching_pipeline.py` | `load_data()` in backtests and pair selection | Continuous OHLCV parquet per symbol. Filename pattern: `{SYMBOL}_{interval}_{start}-{end}.parquet`. Universe membership per month is defined separately in `config/schemas/list_of_assets.json`. **Not on Zenodo** - fetch locally via `data_fetching_pipeline`.                  |
| **`pair_selection/`** | `runners/run_pair_selection.py` | `runners/run_backtest.py` | For each month (`YYYY-MM/`), a ranked table of cointegrated pairs: `pair_selection_{start}_{end}.parquet` (columns include `pair`, `score`, …). **Required before backtest** - if a month is missing, `run_backtest` stops with an error pointing you to `run_pair_selection.py`. |
| **`rl_training/`** | `run_backtest` with `save_for_training=true` | `train_agent` (`rl.training_folder` or first subfolder) | Per-pair `returns_{X}_{Y}_{start}_{end}.parquet` copies of test-window strategy data (z-score, beta, vol, etc.) exported from a specific backtest run. Subfolder name matches that backtest’s `results/run_backtest_<timestamp>/` id.                                             |
| **`rl_models/`** | `train_agent` | `run_backtest` with `performance.use_rl=true` | Stable-Baselines3 model (`.zip`) and `VecNormalize` stats (`_normalize.pkl`). Basename encodes algorithm, observation space, reward, WandB run id, and seed. Pass the basename (without extension) as `rl_model_folder=...`.                                                      |

**Typical order of population:**

1. `data_fetching_pipeline` → `historical/` + `list_of_assets.json`
2. `run_pair_selection` → `pair_selection/` (can take a while; safe to rerun per month)
3. `run_backtest` → reads `historical/` + `pair_selection/`; writes to `results/`
4. `run_backtest` with `save_for_training=true` → also writes `rl_training/<run_id>/`
5. `train_agent` → reads `rl_training/...`, writes `rl_models/`
6. `run_backtest` with `performance.use_rl=true` → loads from `rl_models/`

`data/` is gitignored except placeholder folders (see `.gitignore`). Use `helpers/download_artifacts.py` for `pair_selection/`, `rl_training/`, and `rl_models/`; fetch `historical/` locally via `data_fetching_pipeline`. Monthly universes live in `config/schemas/list_of_assets.json` (committed to this repo).

---

## Configuration (Hydra + Pydantic)

Experiment settings live in YAML under `config/`. You normally **do not** hard-code parameters in Python - you change YAML or pass overrides on the CLI.

We use a **hybrid Hydra + Pydantic** setup:

| Layer | Role |
|--------|------|
| **Hydra** | Loads and composes YAML files (`defaults`), supports CLI overrides and **multirun** (`-m`) sweeps over many parameter combinations. |
| **Pydantic** | Defines the allowed shape of the config in Python (`modules/core/config.py`): types, enums, date ordering, mutually consistent fields. Invalid configs fail **before** a backtest starts. |
| **JSON schemas** | Exported from Pydantic into `config/schemas/`. Your editor can autocomplete YAML keys and show descriptions inline (`# yaml-language-server: $schema=schemas/schema.json` at the top of each config file). |

At runtime, a runner does roughly: `DictConfig` (Hydra) → `Config(**OmegaConf.to_container(cfg, resolve=True))` (Pydantic). That gives you both flexible YAML editing and strict validation.

**After changing** fields in `modules/core/config.py`, regenerate schemas and the human-readable parameter list:

```bash
poetry run python helpers/update_config.py
# or: docker compose run --rm update_config
```

This updates `config/schemas/*.json` and [`docs/configuration.md`](docs/configuration.md) (auto-generated reference of every parameter).

**Key YAML files**

| File | Purpose |
|------|---------|
| `base.yaml` | Shared defaults: fees, interval, WandB project, Hydra output directories. Included by other configs via `defaults: - base`. |
| `run_backtest.yaml` | Pair selection window, z-score thresholds, walk-forward iterations, RL on/off. |
| `train_agent.yaml` | RL reward type, observation space, training data folder; pulls an algorithm preset from `rl_algo/`. |
| `config/helpers/data_fetching_pipeline.yaml` | How many assets to keep, date ranges, whitelist/blacklist for the download pipeline. |

**Examples of CLI overrides** (no file edit needed):

```bash
# Statistical backtest with custom entry threshold
poetry run python runners/run_backtest.py performance.entry_threshold=3.0

# RL backtest using a specific trained run folder under results/
poetry run python runners/run_backtest.py performance.use_rl=true rl_model_folder=train_agent_2024-03-01_12-00-00_abc123
```

For a full parameter glossary, see [`docs/configuration.md`](docs/configuration.md).

---

## Helpers

Helpers are scripts outside the main `runners/` loop. Two groups: **core** (needed to prepare data and keep config in sync) and **analysis** (optional, for paper figures and robustness checks).

### Core (required for reproduction)

| Script | Role |
|--------|------|
| [`helpers/update_config.py`](helpers/update_config.py) | Exports Pydantic models → `config/schemas/*.json`; regenerates [`docs/configuration.md`](docs/configuration.md). Run after clone and after any change to `modules/core/config.py`. |
| [`helpers/data_fetching_pipeline.py`](helpers/data_fetching_pipeline.py) | End-to-end data pipeline: lists all historical USDT-M futures symbols from Binance Data Vision (avoids survivorship bias from “currently listed only”), downloads monthly klines, drops assets with gaps in the pair-selection window, ranks by liquidity, and writes `data/*.parquet` plus `config/schemas/list_of_assets.json` and [`docs/list_of_assets.md`](docs/list_of_assets.md). |

`list_of_assets.json` is required by `run_backtest.py`. For each monthly walk-forward step it defines **which symbols** were tradable and liquid in that period - the same universes used in the paper.

Tune the pipeline in `config/helpers/data_fetching_pipeline.yaml` (`top_n`, `start`/`end`/`test_end`, `iterations`, `whitelist`/`blacklist`).

```bash
poetry run python helpers/data_fetching_pipeline.py
# or: docker compose run --rm data_fetching_pipeline
```

### Analysis scripts (optional)

Located in [`helpers/analysis_scripts/`](helpers/analysis_scripts/). They read finished runs under `results/` and produce aggregated tables, distribution plots, sensitivity reports, bootstrap tests, WandB exports, etc. **You do not need them** to run a backtest or train an agent - only to replicate paper-style post-processing.

> **Note:** These scripts assume the folder names and layout from our experiments (e.g. grouped under `Baseline Optimization/Stage 1/`). If your `results/` tree differs, open the script and adjust the path constants at the top.

Examples:

```bash
poetry run python helpers/analysis_scripts/generate_distributions.py
poetry run python helpers/analysis_scripts/sensitivity_analysis.py
```

Command history and expected `results/` grouping for the study: [`docs/experiments_commands.md`](docs/experiments_commands.md).

---

## Runners

Runners are the main programs to execute. Each has a matching Docker Compose service.

| Runner | Command (Poetry) | Compose service      | What it does |
|--------|------------------|----------------------|--------------|
| Backtest | `poetry run python runners/run_backtest.py` | `run_backtest`       | Walk-forward backtest on many pairs: select cointegrated pairs → test with z-score rules (and optionally an RL agent for exits). |
| RL training | `poetry run python runners/train_agent.py` | `train_agent`        | Trains A2C or Recurrent PPO on saved backtest trajectories under `data/rl_training/`. |
| Pair selection only | `poetry run python runners/run_pair_selection.py` | `run_pair_selection` | Runs only the pair-selection stage (debugging or custom workflows). |

**Statistical backtest (baseline)** - default in `run_backtest.yaml` (`performance.use_rl=false`). For each month: pick pairs from `list_of_assets.json`, estimate hedge ratios, apply entry/exit/stop-loss rules on the z-score. Repeats for `performance.iterations` months.

**RL backtest** - set `performance.use_rl=true`. The same pair selection and data windows apply, but position management can use a trained policy. Point to a specific checkpoint with `rl_model_folder=<name of folder under results/>` (folder name from a previous `train_agent` run).

**RL training** - expects training tensors produced by a backtest with `save_for_training=true` (writes under `data/rl_training/<run_id>/`). Logs metrics to [Weights & Biases](https://wandb.ai); set `WANDB_API_KEY` in `.env` and adjust `wandb.project` / `wandb.mode` in `base.yaml` (`online` vs `offline`).

Typical RL sequence:

```bash
# 1) Backtest that exports RL training data
poetry run python runners/run_backtest.py save_for_training=true performance.use_rl=false

# 2) Train (outputs under results/train_agent_<timestamp>/)
poetry run python runners/train_agent.py

# 3) Evaluate agent in backtest
poetry run python runners/run_backtest.py performance.use_rl=true rl_model_folder=<folder_from_step_2>
```

---

## Results layout

Every **single** run creates its own directory under `results/`. The name encodes the runner and timestamp so runs never overwrite each other.

```
results/run_backtest_2024-01-15_14-30-00_a1b2c3/
├── .hydra/
│   ├── config.yaml      # resolved task config (exact settings used)
│   ├── hydra.yaml
│   └── overrides.yaml   # CLI overrides applied on top of YAML
├── execution.log
├── <pair>/test/         # per-pair outputs (returns, trades, stats, optional plots)
└── ...
```

The `.hydra/` folder is written by [`save_hydra_config_snapshot`](runners/core/utils.py). That makes each result **self-describing**: months later you can see which `entry_threshold`, dates, and flags produced a given folder without guessing from filenames alone.

**Multirun** (parameter grids) uses a different root, configured in `config/base.yaml`:

```
results/multirun/2024-01-15_14-30-00/
├── 0/                   # first combination in the grid
│   └── .hydra/config.yaml
├── 1/
├── ...
```

Each subdirectory is one point in the sweep, with its own `.hydra/config.yaml`. Analysis scripts often expect you to group or rename these folders to match the structure documented in `docs/experiments_commands.md`.

---

## Multirun & hyperparameter grids

To try many parameter values in one command, use Hydra **multirun** mode: add `-m` and list comma-separated values (or `range()` syntax). Hydra builds the full Cartesian product - e.g. three entry thresholds × two stop losses = six runs.

Use the **joblib launcher** to run combinations in parallel on all CPU cores:

```bash
poetry run python runners/run_backtest.py -m \
  hydra/launcher=joblib hydra.launcher.n_jobs=-1 \
  clean_single_backtests=false generate_plots=false \
  performance.entry_threshold=2.0,2.25,2.50,2.75,3.0 \
  performance.exit_threshold=0.0
```

Why we use this stack for grids:

1. **Pydantic** - every combination is validated; typos in YAML or CLI are caught early.
2. **Hydra multirun** - reproducible sweeps without maintaining dozens of nearly identical config files.
3. **hydra-joblib-launcher** - embarrassingly parallel backtests on local hardware (`n_jobs=-1` uses all cores).

For long optimization stages from the paper (stage 1 / stage 2 zoom grids, RL sensitivity, etc.), see the ready-made commands in [`docs/experiments_commands.md`](docs/experiments_commands.md). Copy a block, adjust dates if needed, and run.

---

## Documentation

| Artifact | How to generate | What it contains |
|----------|-----------------|------------------|
| [`docs/configuration.md`](docs/configuration.md) | `poetry run python helpers/update_config.py` | Human-readable list of all config fields and descriptions. |
| [`docs/list_of_assets.md`](docs/list_of_assets.md) | `poetry run python helpers/data_fetching_pipeline.py` | Per-month traded universes and methodology notes. |
| API HTML (`docs/api/`) | `poetry run pdoc ./modules ./helpers ./runners -o ./docs/api` | Browseable HTML docs for Python modules. |
| [`docs/experiments_commands.md`](docs/experiments_commands.md) | *(static, hand-maintained)* | Exact multirun commands and expected `results/` folder layout from the study. |

**Regenerate schemas, config doc, and API in one go:**

```bash
poetry run python helpers/update_config.py
poetry install --with docs
poetry run pdoc ./modules ./helpers ./runners -o ./docs/api
```

Open `docs/api/index.html` in a browser after generation.

---

## Frozen artifacts (Zenodo)

Re-training RL for several hours is not always necessary - especially for reviewers who only need to verify reported metrics or inspect individual backtests from the paper.

**Zenodo record:** [https://zenodo.org/records/20355140](https://zenodo.org/record/20355140)
**DOI:** [10.5281/zenodo.20355140](https://doi.org/10.5281/zenodo.20355140)

The helper script `helpers/download_artifacts.py` downloads both archived trees from Zenodo:

```bash
poetry run python helpers/download_artifacts.py
# or: docker compose run --rm download_artifacts
```

No Python or Docker? Grab [`data.zip`](https://zenodo.org/records/20355140/files/data.zip) and [`results.zip`](https://zenodo.org/records/20355140/files/results.zip) straight from the Zenodo record and extract them in the repo root - see [Fast path](#fast-path) for the manual flow.

| Archive | Contents                                                                                                                                                                                   |
|---------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **`data/`** | `pair_selection/`, `rl_training/`, and `rl_models/`. **Does not include `data/historical/`** (Binance Data Vision OHLCV - fetch locally; see below).                                       |
| **`results/`** | Per-run parquet outputs (returns, trades, stats) and `.hydra/config.yaml` + `.hydra/overrides.yaml`. Does **not** include `execution.log` or `.hydra/hydra.yaml` (local filesystem paths). |

**To inspect:** extract and browse `results/` - nothing else needed.

**To re-run backtests:** after running `helpers/download_artifacts.py`, fetch historical OHLCV (not on Zenodo), then run backtests against the bundled checkpoints:

```bash
poetry run python helpers/data_fetching_pipeline.py
poetry run python runners/run_backtest.py performance.use_rl=true rl_model_folder=<folder_from_zenodo>
# or with Docker:
docker compose run --rm data_fetching_pipeline
docker compose run --rm run_backtest
```

**Why `historical/` is excluded:** OHLCV is sourced from the public [Binance Data Vision](https://data.binance.vision/) archive. We do not redistribute it on Zenodo because of licensing and redistribution concerns; market data remains the intellectual property of Binance. Use `helpers/data_fetching_pipeline.py` to download the same files we used in the study.

---

## Tests & CI

```bash
poetry run pytest
poetry run ruff check .
poetry run black --check .
```

Unit tests cover core indicators, execution helpers, reward functions, and performance statistics under `tests/`. GitHub Actions (`.github/workflows/ci.yml`) runs Ruff, Black, and **pytest** on push/PR to `main` and `develop`.

---

## License

Research and educational use only. Commercial use prohibited. See [LICENSE](LICENSE).

---

**Paper (arXiv):** *DOI / arXiv link - to be added after publication.*

**Frozen artifacts (Zenodo):** [record](https://zenodo.org/record/XXXXXXX) · [DOI 10.5281/zenodo.XXXXXXX](https://doi.org/10.5281/zenodo.XXXXXXX) *(placeholders)*
