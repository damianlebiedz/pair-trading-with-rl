# Pair Trading Research Framework

Repository for the "Dynamic Multi-Pair Trading Strategy in Cryptocurrency Markets with Deep Reinforcement Learning" (Lebiedź and Ślepaczuk, 2026). Implements **statistical arbitrage** backtesting framework and **reinforcement learning** training and evaluation pipelines.

**Stack:** Python 3.12 · Poetry · Hydra · Pydantic · pandas · statsmodels · scikit-optimize · Gymnasium · Stable-Baselines3 · SB3-Contrib (Recurrent PPO) · Weights & Biases · joblib (parallel sweeps) · Docker

---

## Quick start

```bash
git clone https://github.com/damianlebiedz/research-paper.git
cd research-paper
cp .env.example .env   # set WANDB_API_KEY (required for RL training monitoring)

poetry install
poetry run python helpers/update_config.py          # refresh schemas + docs/configuration.md
poetry run python helpers/data_fetching_pipeline.py # fetch data + list_of_assets (long-running)
poetry run python runners/run_backtest.py
```

**Docker** - prepend `docker compose run --rm <service>` to any command; Hydra CLI overrides work the same way:

```bash
docker compose build
docker compose run --rm update_config
docker compose run --rm data_fetching_pipeline
docker compose run --rm run_backtest performance.use_rl=false
docker compose run --rm train_agent
```

Override the default service command when needed (e.g. multirun):

```bash
docker compose run --rm run_backtest python runners/run_backtest.py -m hydra/launcher=joblib hydra.launcher.n_jobs=-1 performance.entry_threshold=2.0,2.5,3.0
```

---

## Project structure

```
.
├── config/                      # Hydra YAML + JSON schemas (IDE autocomplete / validation)
│   ├── base.yaml                # shared defaults (market, wandb, hydra dirs)
│   ├── run_backtest.yaml        # statistical / RL backtest runner
│   ├── train_agent.yaml         # RL training runner
│   ├── rl_algo/                 # A2C / Recurrent PPO presets
│   ├── helpers/                 # helper-specific YAML (data pipeline)
│   └── schemas/                 # generated JSON schemas + list_of_assets.json
├── data/                        # OHLCV parquet (from pipeline or Zenodo)
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

---

## Configuration (Hydra + Pydantic)

YAML files in `config/` are the runtime source of truth. Each top-level config references a JSON Schema (`# yaml-language-server: $schema=schemas/schema.json`) generated from **Pydantic models** in [`modules/core/config.py`](modules/core/config.py).

| Layer | Role |
|--------|------|
| **Hydra** | Compose configs (`defaults`), CLI overrides, **multirun** (`-m`) sweeps |
| **Pydantic** | Strict validation, cross-field rules, enums, descriptions |
| **JSON schemas** | IDE autocomplete, inline docs, YAML validation in the editor |

At runtime, runners load `DictConfig` → `Config(**OmegaConf.to_container(cfg, resolve=True))`. Invalid combinations fail fast before any backtest runs.

**After changing** `modules/core/config.py`, regenerate artifacts:

```bash
poetry run python helpers/update_config.py
# or: docker compose run --rm update_config
```

This updates `config/schemas/*.json` and [`docs/configuration.md`](docs/configuration.md) (auto-generated parameter reference).

**Key YAML files**

| File | Purpose |
|------|---------|
| `base.yaml` | `market`, `settings`, `wandb`, Hydra output dirs |
| `run_backtest.yaml` | pair selection, performance / strategy params |
| `train_agent.yaml` | RL env + training; pulls `rl_algo` preset |
| `config/helpers/data_fetching_pipeline.yaml` | universe size, date windows, whitelist/blacklist |

Override any leaf from CLI, e.g. `performance.entry_threshold=3.0 pair_selection.top_n=20`.

---

## Helpers

### Core (required for reproduction)

| Script | Role |
|--------|------|
| [`helpers/update_config.py`](helpers/update_config.py) | Exports Pydantic → `config/schemas/*.json`; regenerates `docs/configuration.md` |
| [`helpers/data_fetching_pipeline.py`](helpers/data_fetching_pipeline.py) | Downloads Binance Data Vision futures klines (no survivorship bias), validates gaps, writes `data/*.parquet`, `config/schemas/list_of_assets.json`, [`docs/list_of_assets.md`](docs/list_of_assets.md) |

`list_of_assets.json` is consumed by `run_backtest.py` — each monthly iteration maps to a liquidity-ranked universe used in pair selection.

Configure the pipeline via `config/helpers/data_fetching_pipeline.yaml` (dates, `top_n`, iterations, whitelist/blacklist).

### Analysis scripts (optional)

Located in [`helpers/analysis_scripts/`](helpers/analysis_scripts/). They aggregate backtest / RL / WandB outputs into tables and plots for the thesis (distributions, sensitivity, IS/OOS, seed variance, bootstrap, etc.).

> **Note:** These scripts were written for the exact folder layout and experiment names used in this paper. For different `results/` layouts or naming, expect to adjust paths/constants inside each script.

Examples:

```bash
poetry run python helpers/analysis_scripts/generate_distributions.py
poetry run python helpers/analysis_scripts/sensitivity_analysis.py
```

See [`docs/experiments_commands.md`](docs/experiments_commands.md) for the multirun commands used during the study and expected `results/` grouping.

---

## Runners

| Runner | Command (Poetry) | Compose service |
|--------|------------------|-----------------|
| Backtest | `poetry run python runners/run_backtest.py` | `run_backtest` |
| RL training | `poetry run python runners/train_agent.py` | `train_agent` |
| Pair selection only | `poetry run python runners/run_pair_selection.py` | — |

**RL training** logs to [Weights & Biases](https://wandb.ai) (`wandb.project`, `wandb.mode` in `base.yaml`). Set `WANDB_API_KEY` in `.env`.

**Statistical backtest** → walk-forward: pair selection → optimization window → test window, repeated monthly (`performance.iterations`).

**RL backtest** → set `performance.use_rl=true` and optionally `rl_model_folder=<run_folder_name>`.

---

## Results layout

Each single run creates a timestamped directory:

```
results/run_backtest_2024-01-15_14-30-00_a1b2c3/
├── .hydra/
│   ├── config.yaml      # resolved task config
│   ├── hydra.yaml
│   └── overrides.yaml
├── execution.log
├── <pair>/test/         # per-pair outputs
└── ...
```

`.hydra/` is written by [`save_hydra_config_snapshot`](runners/core/utils.py) so every result folder is self-describing for later analysis scripts.

**Multirun** sweeps go to `results/multirun/<timestamp>/` (see `hydra.sweep.dir` in `config/base.yaml`). Each grid point is a subdirectory with its own `.hydra/config.yaml`.

---

## Multirun & hyperparameter grids

Hydra `-m` expands comma-separated or `range()` overrides into a Cartesian grid. Use the **joblib launcher** for parallel execution:

```bash
poetry run python runners/run_backtest.py -m \
  hydra/launcher=joblib hydra.launcher.n_jobs=-1 \
  clean_single_backtests=false generate_plots=false \
  performance.entry_threshold=2.0,2.25,2.50,2.75,3.0 \
  performance.exit_threshold=0.0
```

Why this setup works well:

1. **Pydantic** validates every combination before execution.
2. **Hydra multirun** gives a reproducible grid without copy-pasting YAML.
3. **hydra-joblib-launcher** runs independent jobs in parallel (`n_jobs=-1` → all cores).

Full command history for the paper: [`docs/experiments_commands.md`](docs/experiments_commands.md).

---

## Documentation

| Artifact | How to generate |
|----------|-----------------|
| [`docs/configuration.md`](docs/configuration.md) | `poetry run python helpers/update_config.py` |
| [`docs/list_of_assets.md`](docs/list_of_assets.md) | `poetry run python helpers/data_fetching_pipeline.py` |
| API HTML (`docs/api/`) | `poetry run pdoc ./modules ./helpers ./runners -o ./docs/api` |

**Regenerate everything (schemas + config doc + API):**

```bash
poetry run python helpers/update_config.py
poetry run pdoc ./modules ./helpers ./runners -o ./docs/api
```

Install doc dependencies: `poetry install --with docs`.

Static reference (not auto-generated): [`docs/experiments_commands.md`](docs/experiments_commands.md).

---

## Frozen artifacts (Zenodo)

For reviewers who should not re-fetch data or re-train RL models, use the Makefile (set Zenodo URLs first):

```bash
make download_artifacts
```

Then run backtests against extracted `data/` and `results/` (pre-trained models).

---

## Tests & CI

```bash
poetry run pytest
poetry run ruff check .
poetry run black --check .
```

GitHub Actions (`.github/workflows/ci.yml`) runs Ruff, Black, and **pytest** on push/PR to `main` and `develop`.

---

## License

Research and educational use only. Commercial use prohibited. See [LICENSE](LICENSE).
