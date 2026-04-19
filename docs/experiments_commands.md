# Experiments Commands

- [Baseline Optimization](#baseline-optimization)
  - [STAGE 1](#stage-1)
  - [STAGE 1 3.0 ZOOM](#stage-1-30-zoom)
  - [STAGE 1 3.5 ZOOM](#stage-1-35-zoom)
  - [STAGE 2](#stage-2)
  - [STAGE 2 1.5 ZOOM](#stage-2-15-zoom)
  - [STAGE 2 2.0 ZOOM](#stage-2-20-zoom)
- [Baseline Sensitivity Analysis](#baseline-sensitivity-analysis)
  - [WIDE](#wide)
  - [MICRO](#micro)
  - [ASSUMPTIONS VERIFICATION](#assumptions-verification)
- [RL Training](#rl-training)
  - [TRAINING](#training)
- [RL Test](#rl-test)
- [RL OOS Sensitivity Analysis](#rl-oos-sensitivity-analysis)
  - [WIDE](#wide)
  - [MICRO](#micro)
  - [ASSUMPTIONS VERIFICATION](#assumptions-verification)

Important: All commands require default values in the config `*.yaml` files.

## Baseline Optimization

Note: Folders should be grouped in this structure:
```
.
├── results/
│   ├── Baseline Optimization/
│   │   ├── Stage 1/
│   │   │   ├── run_backtest_*/
│   │   │   ├── .../
│   │   ├── Stage 1 zoom 3.0/
│   │   │   ├── run_backtest_*/
│   │   │   ├── .../
│   │   ├── .../
```
Only then the `generate_distributions.py` helper will be able to generate report in `Baseline Optimization/distributions`.

### STAGE 1

```bash
poetry run python runners/run_backtest.py -m hydra/launcher=joblib hydra.launcher.n_jobs=-1 clean_single_backtests=false generate_plots=false save_for_training=false performance.use_rl=false performance.iterations=12 performance.beta_hedge=rolling performance.sl_lock=true performance.time_decay_sl=true pair_selection.top_n=20 pair_selection.start="2023-11-01" pair_selection.end="2024-01-01" performance.beta_start="2023-12-01" performance.start="2024-01-01" performance.end="2024-02-01" performance.entry_threshold=2.00,2.25,2.50,2.75,3.00,3.25,3.50,3.75,4.00 performance.exit_threshold=0.0 performance.stop_loss=null performance.z_score_window=168;
```

### STAGE 1 3.0 ZOOM

```bash
poetry run python runners/run_backtest.py -m hydra/launcher=joblib hydra.launcher.n_jobs=-1 clean_single_backtests=false generate_plots=false save_for_training=false performance.use_rl=false performance.iterations=12 performance.beta_hedge=rolling performance.sl_lock=true performance.time_decay_sl=true pair_selection.top_n=20 pair_selection.start="2023-11-01" pair_selection.end="2024-01-01" performance.beta_start="2023-12-01" performance.start="2024-01-01" performance.end="2024-02-01" performance.entry_threshold=3.25,3.30,3.35,3.40,3.45,3.50,3.65,3.70,3.75 performance.exit_threshold=0.0 performance.stop_loss=null performance.z_score_window=168;
```

### STAGE 1 3.5 ZOOM

```bash
poetry run python runners/run_backtest.py -m hydra/launcher=joblib hydra.launcher.n_jobs=-1 clean_single_backtests=false generate_plots=false save_for_training=false performance.use_rl=false performance.iterations=12 performance.beta_hedge=rolling performance.sl_lock=true performance.time_decay_sl=true pair_selection.top_n=20 pair_selection.start="2023-11-01" pair_selection.end="2024-01-01" performance.beta_start="2023-12-01" performance.start="2024-01-01" performance.end="2024-02-01" performance.entry_threshold=2.75,2.80,2.95,3.00,3.05,3.01,3.15,3.20,3.25 performance.exit_threshold=0.0 performance.stop_loss=null performance.z_score_window=168;
```

### STAGE 2

```bash
poetry run python runners/run_backtest.py -m hydra/launcher=joblib hydra.launcher.n_jobs=-1 clean_single_backtests=false generate_plots=false save_for_training=false performance.use_rl=false performance.iterations=12 performance.beta_hedge=rolling performance.sl_lock=true performance.time_decay_sl=true pair_selection.top_n=20 pair_selection.start="2023-11-01" pair_selection.end="2024-01-01" performance.beta_start="2023-12-01" performance.start="2024-01-01" performance.end="2024-02-01" performance.entry_threshold=3.00 performance.exit_threshold=0.0 performance.stop_loss=1.25,1.50,1.75,2.00,2.25,2.50,2.75,3.00,3.25 performance.z_score_window=168;
```

### STAGE 2 1.5 ZOOM

```bash
poetry run python runners/run_backtest.py -m hydra/launcher=joblib hydra.launcher.n_jobs=-1 clean_single_backtests=false generate_plots=false save_for_training=false performance.use_rl=false performance.iterations=12 performance.beta_hedge=rolling performance.sl_lock=true performance.time_decay_sl=true pair_selection.top_n=20 pair_selection.start="2023-11-01" pair_selection.end="2024-01-01" performance.beta_start="2023-12-01" performance.start="2024-01-01" performance.end="2024-02-01" performance.entry_threshold=3.00 performance.exit_threshold=0.0 performance.stop_loss=1.25,1.30,1.35,1.40,1.45,1.50,1.55,1.60,1.65,1.70,1.75 performance.z_score_window=168;
```

### STAGE 2 2.0 ZOOM

```bash
poetry run python runners/run_backtest.py -m hydra/launcher=joblib hydra.launcher.n_jobs=-1 clean_single_backtests=false generate_plots=false save_for_training=false performance.use_rl=false performance.iterations=12 performance.beta_hedge=rolling performance.sl_lock=true performance.time_decay_sl=true pair_selection.top_n=20 pair_selection.start="2023-11-01" pair_selection.end="2024-01-01" performance.beta_start="2023-12-01" performance.start="2024-01-01" performance.end="2024-02-01" performance.entry_threshold=3.00 performance.exit_threshold=0.0 performance.stop_loss=1.75,1.80,1.85,1.90,1.95,2.00,2.05,2.10,2.15,2.20,2.25 performance.z_score_window=168;
```

## Baseline Sensitivity Analysis

Note: Folders should be grouped in this structure:
```
.
├── results/
│   ├── Baseline Sensitivity Analysis/
│   │   ├── Assumptions Verification/
│   │   │   ├── is/
│   │   │   │   ├── baseline_is*/
│   │   │   │   ├── run_backtest_*/
│   │   │   │   ├── .../
│   │   │   ├── oos/
│   │   │   │   ├── baseline_oos*/
│   │   │   │   ├── .../
│   │   ├── Micro/
│   │   │   ├── .../
│   │   ├── Macro/
│   │   │   ├── .../
```
Important: `baseline_is` and `baseline_oos` are crucial - copy the appropriate backtest folders into these directories. 
Only then the `sensitivity_analysis.py` helper will be able to generate report in `Baseline Sensitivity Analysis/*/*_report`.
Remember to set a specific `FOLDER` name before every execution.

### WIDE

#### IS

```bash
poetry run python runners/run_backtest.py -m hydra/launcher=joblib hydra.launcher.n_jobs=-1 clean_single_backtests=false generate_plots=false save_for_training=false performance.use_rl=false performance.iterations=12 performance.beta_hedge=rolling performance.sl_lock=true performance.time_decay_sl=true pair_selection.top_n=20 pair_selection.start="2023-11-01" pair_selection.end="2024-01-01" performance.beta_start="2023-12-01" performance.start="2024-01-01" performance.end="2024-02-01" performance.entry_threshold=2.50,3.50 performance.exit_threshold=0.0 performance.stop_loss=2.0 performance.z_score_window=168;
poetry run python runners/run_backtest.py -m hydra/launcher=joblib hydra.launcher.n_jobs=-1 clean_single_backtests=false generate_plots=false save_for_training=false performance.use_rl=false performance.iterations=12 performance.beta_hedge=rolling performance.sl_lock=true performance.time_decay_sl=true pair_selection.top_n=20 pair_selection.start="2023-11-01" pair_selection.end="2024-01-01" performance.beta_start="2023-12-01" performance.start="2024-01-01" performance.end="2024-02-01" performance.entry_threshold=3.0 performance.exit_threshold=-0.5,0.5 performance.stop_loss=2.0 performance.z_score_window=168;
poetry run python runners/run_backtest.py -m hydra/launcher=joblib hydra.launcher.n_jobs=-1 clean_single_backtests=false generate_plots=false save_for_training=false performance.use_rl=false performance.iterations=12 performance.beta_hedge=rolling performance.sl_lock=true performance.time_decay_sl=true pair_selection.top_n=20 pair_selection.start="2023-11-01" pair_selection.end="2024-01-01" performance.beta_start="2023-12-01" performance.start="2024-01-01" performance.end="2024-02-01" performance.entry_threshold=3.0 performance.exit_threshold=0.0 performance.stop_loss=1.5,2.5 performance.z_score_window=168;
poetry run python runners/run_backtest.py -m hydra/launcher=joblib hydra.launcher.n_jobs=-1 clean_single_backtests=false generate_plots=false save_for_training=false performance.use_rl=false performance.iterations=12 performance.beta_hedge=rolling performance.sl_lock=true performance.time_decay_sl=true pair_selection.top_n=20 pair_selection.start="2023-11-01" pair_selection.end="2024-01-01" performance.beta_start="2023-12-01" performance.start="2024-01-01" performance.end="2024-02-01" performance.entry_threshold=3.0 performance.exit_threshold=0.0 performance.stop_loss=2.0 performance.z_score_window=168 pair_selection.top_n=10,30;
poetry run python runners/run_backtest.py -m hydra/launcher=joblib hydra.launcher.n_jobs=-1 clean_single_backtests=false generate_plots=false save_for_training=false performance.use_rl=false performance.iterations=12 performance.beta_hedge=rolling performance.sl_lock=true performance.time_decay_sl=true pair_selection.top_n=20 pair_selection.start="2023-11-01" pair_selection.end="2024-01-01" performance.beta_start="2023-12-01" performance.start="2024-01-01" performance.end="2024-02-01" performance.entry_threshold=3.0 performance.exit_threshold=0.0 performance.stop_loss=2.0 performance.z_score_window=120,216 pair_selection.top_n=20;
```

#### OOS
    
```bash
poetry run python runners/run_backtest.py -m hydra/launcher=joblib hydra.launcher.n_jobs=-1 clean_single_backtests=false generate_plots=false save_for_training=false performance.use_rl=false performance.iterations=12 performance.beta_hedge=rolling performance.sl_lock=true performance.time_decay_sl=true pair_selection.top_n=20 pair_selection.start="2024-11-01" pair_selection.end="2025-01-01" performance.beta_start="2024-12-01" performance.start="2025-01-01" performance.end="2025-02-01" performance.entry_threshold=2.50,3.50 performance.exit_threshold=0.0 performance.stop_loss=2.0 performance.z_score_window=168;
poetry run python runners/run_backtest.py -m hydra/launcher=joblib hydra.launcher.n_jobs=-1 clean_single_backtests=false generate_plots=false save_for_training=false performance.use_rl=false performance.iterations=12 performance.beta_hedge=rolling performance.sl_lock=true performance.time_decay_sl=true pair_selection.top_n=20 pair_selection.start="2024-11-01" pair_selection.end="2025-01-01" performance.beta_start="2024-12-01" performance.start="2025-01-01" performance.end="2025-02-01" performance.entry_threshold=3.0 performance.exit_threshold=-0.5,0.5 performance.stop_loss=2.0 performance.z_score_window=168;
poetry run python runners/run_backtest.py -m hydra/launcher=joblib hydra.launcher.n_jobs=-1 clean_single_backtests=false generate_plots=false save_for_training=false performance.use_rl=false performance.iterations=12 performance.beta_hedge=rolling performance.sl_lock=true performance.time_decay_sl=true pair_selection.top_n=20 pair_selection.start="2024-11-01" pair_selection.end="2025-01-01" performance.beta_start="2024-12-01" performance.start="2025-01-01" performance.end="2025-02-01" performance.entry_threshold=3.0 performance.exit_threshold=0.0 performance.stop_loss=1.5,2.5 performance.z_score_window=168;
poetry run python runners/run_backtest.py -m hydra/launcher=joblib hydra.launcher.n_jobs=-1 clean_single_backtests=false generate_plots=false save_for_training=false performance.use_rl=false performance.iterations=12 performance.beta_hedge=rolling performance.sl_lock=true performance.time_decay_sl=true pair_selection.top_n=20 pair_selection.start="2024-11-01" pair_selection.end="2025-01-01" performance.beta_start="2024-12-01" performance.start="2025-01-01" performance.end="2025-02-01" performance.entry_threshold=3.0 performance.exit_threshold=0.0 performance.stop_loss=2.0 performance.z_score_window=168 pair_selection.top_n=10,30;
poetry run python runners/run_backtest.py -m hydra/launcher=joblib hydra.launcher.n_jobs=-1 clean_single_backtests=false generate_plots=false save_for_training=false performance.use_rl=false performance.iterations=12 performance.beta_hedge=rolling performance.sl_lock=true performance.time_decay_sl=true pair_selection.top_n=20 pair_selection.start="2024-11-01" pair_selection.end="2025-01-01" performance.beta_start="2024-12-01" performance.start="2025-01-01" performance.end="2025-02-01" performance.entry_threshold=3.0 performance.exit_threshold=0.0 performance.stop_loss=2.0 performance.z_score_window=120,216 pair_selection.top_n=20;
```

### MICRO

#### IS

```bash
poetry run python runners/run_backtest.py -m hydra/launcher=joblib hydra.launcher.n_jobs=-1 clean_single_backtests=false generate_plots=false save_for_training=false performance.use_rl=false performance.iterations=12 performance.beta_hedge=rolling performance.sl_lock=true performance.time_decay_sl=true pair_selection.top_n=20 pair_selection.start="2023-11-01" pair_selection.end="2024-01-01" performance.beta_start="2023-12-01" performance.start="2024-01-01" performance.end="2024-02-01" performance.entry_threshold=2.90,3.10 performance.exit_threshold=0.0 performance.stop_loss=2.0 performance.z_score_window=168;
poetry run python runners/run_backtest.py -m hydra/launcher=joblib hydra.launcher.n_jobs=-1 clean_single_backtests=false generate_plots=false save_for_training=false performance.use_rl=false performance.iterations=12 performance.beta_hedge=rolling performance.sl_lock=true performance.time_decay_sl=true pair_selection.top_n=20 pair_selection.start="2023-11-01" pair_selection.end="2024-01-01" performance.beta_start="2023-12-01" performance.start="2024-01-01" performance.end="2024-02-01" performance.entry_threshold=3.0 performance.exit_threshold=-0.1,0.1 performance.stop_loss=2.0 performance.z_score_window=168;
poetry run python runners/run_backtest.py -m hydra/launcher=joblib hydra.launcher.n_jobs=-1 clean_single_backtests=false generate_plots=false save_for_training=false performance.use_rl=false performance.iterations=12 performance.beta_hedge=rolling performance.sl_lock=true performance.time_decay_sl=true pair_selection.top_n=20 pair_selection.start="2023-11-01" pair_selection.end="2024-01-01" performance.beta_start="2023-12-01" performance.start="2024-01-01" performance.end="2024-02-01" performance.entry_threshold=3.0 performance.exit_threshold=0.0 performance.stop_loss=1.9,2.1 performance.z_score_window=168;
poetry run python runners/run_backtest.py -m hydra/launcher=joblib hydra.launcher.n_jobs=-1 clean_single_backtests=false generate_plots=false save_for_training=false performance.use_rl=false performance.iterations=12 performance.beta_hedge=rolling performance.sl_lock=true performance.time_decay_sl=true pair_selection.top_n=20 pair_selection.start="2023-11-01" pair_selection.end="2024-01-01" performance.beta_start="2023-12-01" performance.start="2024-01-01" performance.end="2024-02-01" performance.entry_threshold=3.0 performance.exit_threshold=0.0 performance.stop_loss=2.0 performance.z_score_window=168 pair_selection.top_n=18,22;
poetry run python runners/run_backtest.py -m hydra/launcher=joblib hydra.launcher.n_jobs=-1 clean_single_backtests=false generate_plots=false save_for_training=false performance.use_rl=false performance.iterations=12 performance.beta_hedge=rolling performance.sl_lock=true performance.time_decay_sl=true pair_selection.top_n=20 pair_selection.start="2023-11-01" pair_selection.end="2024-01-01" performance.beta_start="2023-12-01" performance.start="2024-01-01" performance.end="2024-02-01" performance.entry_threshold=3.0 performance.exit_threshold=0.0 performance.stop_loss=2.0 performance.z_score_window=162,174 pair_selection.top_n=20;
```

#### OOS

```bash
poetry run python runners/run_backtest.py -m hydra/launcher=joblib hydra.launcher.n_jobs=-1 clean_single_backtests=false generate_plots=false save_for_training=false performance.use_rl=false performance.iterations=12 performance.beta_hedge=rolling performance.sl_lock=true performance.time_decay_sl=true pair_selection.top_n=20 pair_selection.start="2024-11-01" pair_selection.end="2025-01-01" performance.beta_start="2024-12-01" performance.start="2025-01-01" performance.end="2025-02-01" performance.entry_threshold=2.90,3.10 performance.exit_threshold=0.0 performance.stop_loss=2.0 performance.z_score_window=168;
poetry run python runners/run_backtest.py -m hydra/launcher=joblib hydra.launcher.n_jobs=-1 clean_single_backtests=false generate_plots=false save_for_training=false performance.use_rl=false performance.iterations=12 performance.beta_hedge=rolling performance.sl_lock=true performance.time_decay_sl=true pair_selection.top_n=20 pair_selection.start="2024-11-01" pair_selection.end="2025-01-01" performance.beta_start="2024-12-01" performance.start="2025-01-01" performance.end="2025-02-01" performance.entry_threshold=3.0 performance.exit_threshold=-0.1,0.1 performance.stop_loss=2.0 performance.z_score_window=168;
poetry run python runners/run_backtest.py -m hydra/launcher=joblib hydra.launcher.n_jobs=-1 clean_single_backtests=false generate_plots=false save_for_training=false performance.use_rl=false performance.iterations=12 performance.beta_hedge=rolling performance.sl_lock=true performance.time_decay_sl=true pair_selection.top_n=20 pair_selection.start="2024-11-01" pair_selection.end="2025-01-01" performance.beta_start="2024-12-01" performance.start="2025-01-01" performance.end="2025-02-01" performance.entry_threshold=3.0 performance.exit_threshold=0.0 performance.stop_loss=1.9,2.1 performance.z_score_window=168;
poetry run python runners/run_backtest.py -m hydra/launcher=joblib hydra.launcher.n_jobs=-1 clean_single_backtests=false generate_plots=false save_for_training=false performance.use_rl=false performance.iterations=12 performance.beta_hedge=rolling performance.sl_lock=true performance.time_decay_sl=true pair_selection.top_n=20 pair_selection.start="2024-11-01" pair_selection.end="2025-01-01" performance.beta_start="2024-12-01" performance.start="2025-01-01" performance.end="2025-02-01" performance.entry_threshold=3.0 performance.exit_threshold=0.0 performance.stop_loss=2.0 performance.z_score_window=168 pair_selection.top_n=18,22;
poetry run python runners/run_backtest.py -m hydra/launcher=joblib hydra.launcher.n_jobs=-1 clean_single_backtests=false generate_plots=false save_for_training=false performance.use_rl=false performance.iterations=12 performance.beta_hedge=rolling performance.sl_lock=true performance.time_decay_sl=true pair_selection.top_n=20 pair_selection.start="2024-11-01" pair_selection.end="2025-01-01" performance.beta_start="2024-12-01" performance.start="2025-01-01" performance.end="2025-02-01" performance.entry_threshold=3.0 performance.exit_threshold=0.0 performance.stop_loss=2.0 performance.z_score_window=162,174 pair_selection.top_n=20;
```

### ASSUMPTIONS VERIFICATION

#### IS

```bash
poetry run python runners/run_backtest.py -m hydra/launcher=joblib hydra.launcher.n_jobs=-1 clean_single_backtests=false generate_plots=false save_for_training=false performance.use_rl=false performance.iterations=12 performance.beta_hedge=rolling performance.sl_lock=true performance.time_decay_sl=true pair_selection.top_n=20 pair_selection.start="2023-11-01" pair_selection.end="2024-01-01" performance.beta_start="2023-12-01" performance.start="2024-01-01" performance.end="2024-02-01" performance.entry_threshold=3.0 performance.exit_threshold=0.0 performance.stop_loss=2.0 performance.z_score_window=168 market.fee_rate=0,0.001;
poetry run python runners/run_backtest.py -m hydra/launcher=joblib hydra.launcher.n_jobs=-1 clean_single_backtests=false generate_plots=false save_for_training=false performance.use_rl=false performance.iterations=12 performance.beta_hedge=rolling performance.sl_lock=true performance.time_decay_sl=true pair_selection.top_n=20 pair_selection.start="2023-11-01" pair_selection.end="2024-01-01" performance.beta_start="2023-12-01" performance.start="2024-01-01" performance.end="2024-02-01" performance.entry_threshold=3.0 performance.exit_threshold=0.0 performance.stop_loss=2.0 performance.z_score_window=168 performance.beta_hedge=no_hedge;
poetry run python runners/run_backtest.py -m hydra/launcher=joblib hydra.launcher.n_jobs=-1 clean_single_backtests=false generate_plots=false save_for_training=false performance.use_rl=false performance.iterations=12 performance.beta_hedge=rolling performance.sl_lock=true performance.time_decay_sl=true pair_selection.top_n=20 pair_selection.start="2023-11-01" pair_selection.end="2024-01-01" performance.beta_start="2023-12-01" performance.start="2024-01-01" performance.end="2024-02-01" performance.entry_threshold=3.0 performance.exit_threshold=0.0 performance.stop_loss=2.0 performance.z_score_window=168 performance.sl_lock=false;
poetry run python runners/run_backtest.py -m hydra/launcher=joblib hydra.launcher.n_jobs=-1 clean_single_backtests=false generate_plots=false save_for_training=false performance.use_rl=false performance.iterations=12 performance.beta_hedge=rolling performance.sl_lock=true performance.time_decay_sl=true pair_selection.top_n=20 pair_selection.start="2023-11-01" pair_selection.end="2024-01-01" performance.beta_start="2023-12-01" performance.start="2024-01-01" performance.end="2024-02-01" performance.entry_threshold=3.0 performance.exit_threshold=0.0 performance.stop_loss=2.0 performance.z_score_window=168 performance.time_decay_sl=false;
```

#### OOS

```bash
poetry run python runners/run_backtest.py -m hydra/launcher=joblib hydra.launcher.n_jobs=-1 clean_single_backtests=false generate_plots=false save_for_training=false performance.use_rl=false performance.iterations=12 performance.beta_hedge=rolling performance.sl_lock=true performance.time_decay_sl=true pair_selection.top_n=20 pair_selection.start="2024-11-01" pair_selection.end="2025-01-01" performance.beta_start="2024-12-01" performance.start="2025-01-01" performance.end="2025-02-01" performance.entry_threshold=3.0 performance.exit_threshold=0.0 performance.stop_loss=2.0 performance.z_score_window=168 market.fee_rate=0,0.001;
poetry run python runners/run_backtest.py -m hydra/launcher=joblib hydra.launcher.n_jobs=-1 clean_single_backtests=false generate_plots=false save_for_training=false performance.use_rl=false performance.iterations=12 performance.beta_hedge=rolling performance.sl_lock=true performance.time_decay_sl=true pair_selection.top_n=20 pair_selection.start="2024-11-01" pair_selection.end="2025-01-01" performance.beta_start="2024-12-01" performance.start="2025-01-01" performance.end="2025-02-01" performance.entry_threshold=3.0 performance.exit_threshold=0.0 performance.stop_loss=2.0 performance.z_score_window=168 performance.beta_hedge=no_hedge;
poetry run python runners/run_backtest.py -m hydra/launcher=joblib hydra.launcher.n_jobs=-1 clean_single_backtests=false generate_plots=false save_for_training=false performance.use_rl=false performance.iterations=12 performance.beta_hedge=rolling performance.sl_lock=true performance.time_decay_sl=true pair_selection.top_n=20 pair_selection.start="2024-11-01" pair_selection.end="2025-01-01" performance.beta_start="2024-12-01" performance.start="2025-01-01" performance.end="2025-02-01" performance.entry_threshold=3.0 performance.exit_threshold=0.0 performance.stop_loss=2.0 performance.z_score_window=168 performance.sl_lock=false;
poetry run python runners/run_backtest.py -m hydra/launcher=joblib hydra.launcher.n_jobs=-1 clean_single_backtests=false generate_plots=false save_for_training=false performance.use_rl=false performance.iterations=12 performance.beta_hedge=rolling performance.sl_lock=true performance.time_decay_sl=true pair_selection.top_n=20 pair_selection.start="2024-11-01" pair_selection.end="2025-01-01" performance.beta_start="2024-12-01" performance.start="2025-01-01" performance.end="2025-02-01" performance.entry_threshold=3.0 performance.exit_threshold=0.0 performance.stop_loss=2.0 performance.z_score_window=168 performance.time_decay_sl=false;
```

## RL Training

Note: Before training run In-Sample Baseline's backtest with `save_for_training: false` and ensure that before this action `data/rl_training` folder was clean.

```bash
poetry run python runners/run_backtest.py -m hydra/launcher=joblib hydra.launcher.n_jobs=-1 clean_single_backtests=false generate_plots=false save_for_training=true performance.use_rl=false performance.iterations=12 performance.beta_hedge=rolling performance.sl_lock=true performance.time_decay_sl=true pair_selection.top_n=20 pair_selection.start="2023-11-01" pair_selection.end="2024-01-01" performance.beta_start="2023-12-01" performance.start="2024-01-01" performance.end="2024-02-01" performance.entry_threshold=3.0 performance.exit_threshold=0.0 performance.stop_loss=2.0 performance.z_score_window=168;
```

### TRAINING

```bash
poetry run python runners/train_agent.py -m hydra/launcher=joblib hydra.launcher.n_jobs=-1 defaults.rl_algo=recurrent_ppo rl.training_folder=null rl.reward=StepPnLReward,TradePnLReward,HybridActionReward rl.reward_lambda=1.0,1.2 rl.fee_multiplier=0.2 rl.obs_space_type=autonomous,standard,full rl.passes_per_pair=20 rl.seed=42 rl.verbose=1 rl.time_decay_stop=true
```

## RL Test

Note: Ensurer that you have RL models trained in `data/rl_models`.

### IS

```bash
poetry run python runners/run_backtest.py -m hydra/launcher=joblib hydra.launcher.n_jobs=-1 clean_single_backtests=false generate_plots=false save_for_training=false rl_model_folder="recurrent_ppo_*_*_*_*_seed*" performance.use_rl=true  performance.autonomous_agent=false performance.iterations=12 performance.beta_hedge=rolling performance.sl_lock=true performance.time_decay_sl=true pair_selection.top_n=20 pair_selection.start="2023-11-01" pair_selection.end="2024-01-01" performance.beta_start="2023-12-01" performance.start="2024-01-01" performance.end="2024-02-01" performance.entry_threshold=3.0 performance.exit_threshold=0.0 performance.stop_loss=2.0 performance.z_score_window=168;
```

### OOS

```bash
poetry run python runners/run_backtest.py -m hydra/launcher=joblib hydra.launcher.n_jobs=-1 clean_single_backtests=false generate_plots=false save_for_training=false rl_model_folder="recurrent_ppo_*_*_*_*_seed*" performance.use_rl=true  performance.autonomous_agent=false performance.iterations=12 performance.beta_hedge=rolling performance.sl_lock=true performance.time_decay_sl=true pair_selection.top_n=20 pair_selection.start="2024-11-01" pair_selection.end="2025-01-01" performance.beta_start="2024-12-01" performance.start="2025-01-01" performance.end="2025-02-01" performance.entry_threshold=3.0 performance.exit_threshold=0.0 performance.stop_loss=2.0 performance.z_score_window=168;
```

Repeat this commands for every RL model. Then you can aggregate these backtests into one folder and run helper `aggregate_rl_results.py` with appropriate `FOLDER` name to generate comparison table.

## RL OOS Sensitivity Analysis

Note: Structure of the folder should be similar as in the baseline's case. Remember to change `FOLDER` name in the `sensitivity_analysis.py` file.

### WIDE

```bash
poetry run python runners/run_backtest.py -m hydra/launcher=joblib hydra.launcher.n_jobs=-1 clean_single_backtests=false generate_plots=false save_for_training=false rl_model_folder="recurrent_ppo_*_*_*_*_seed*" performance.use_rl=true  performance.autonomous_agent=false performance.iterations=12 performance.beta_hedge=rolling performance.sl_lock=true performance.time_decay_sl=true pair_selection.top_n=20 pair_selection.start="2024-11-01" pair_selection.end="2025-01-01" performance.beta_start="2024-12-01" performance.start="2025-01-01" performance.end="2025-02-01" performance.entry_threshold=2.50,3.50 performance.exit_threshold=0.0 performance.stop_loss=2.0 performance.z_score_window=168;
poetry run python runners/run_backtest.py -m hydra/launcher=joblib hydra.launcher.n_jobs=-1 clean_single_backtests=false generate_plots=false save_for_training=false rl_model_folder="recurrent_ppo_*_*_*_*_seed*" performance.use_rl=true  performance.autonomous_agent=false performance.iterations=12 performance.beta_hedge=rolling performance.sl_lock=true performance.time_decay_sl=true pair_selection.top_n=20 pair_selection.start="2024-11-01" pair_selection.end="2025-01-01" performance.beta_start="2024-12-01" performance.start="2025-01-01" performance.end="2025-02-01" performance.entry_threshold=3.0 performance.exit_threshold=-0.5,0.5 performance.stop_loss=2.0 performance.z_score_window=168;
poetry run python runners/run_backtest.py -m hydra/launcher=joblib hydra.launcher.n_jobs=-1 clean_single_backtests=false generate_plots=false save_for_training=false rl_model_folder="recurrent_ppo_*_*_*_*_seed*" performance.use_rl=true  performance.autonomous_agent=false performance.iterations=12 performance.beta_hedge=rolling performance.sl_lock=true performance.time_decay_sl=true pair_selection.top_n=20 pair_selection.start="2024-11-01" pair_selection.end="2025-01-01" performance.beta_start="2024-12-01" performance.start="2025-01-01" performance.end="2025-02-01" performance.entry_threshold=3.0 performance.exit_threshold=0.0 performance.stop_loss=1.5,2.5 performance.z_score_window=168;
poetry run python runners/run_backtest.py -m hydra/launcher=joblib hydra.launcher.n_jobs=-1 clean_single_backtests=false generate_plots=false save_for_training=false rl_model_folder="recurrent_ppo_*_*_*_*_seed*" performance.use_rl=true  performance.autonomous_agent=false performance.iterations=12 performance.beta_hedge=rolling performance.sl_lock=true performance.time_decay_sl=true pair_selection.top_n=20 pair_selection.start="2024-11-01" pair_selection.end="2025-01-01" performance.beta_start="2024-12-01" performance.start="2025-01-01" performance.end="2025-02-01" performance.entry_threshold=3.0 performance.exit_threshold=0.0 performance.stop_loss=2.0 performance.z_score_window=168 pair_selection.top_n=10,30;
poetry run python runners/run_backtest.py -m hydra/launcher=joblib hydra.launcher.n_jobs=-1 clean_single_backtests=false generate_plots=false save_for_training=false rl_model_folder="recurrent_ppo_*_*_*_*_seed*" performance.use_rl=true  performance.autonomous_agent=false performance.iterations=12 performance.beta_hedge=rolling performance.sl_lock=true performance.time_decay_sl=true pair_selection.top_n=20 pair_selection.start="2024-11-01" pair_selection.end="2025-01-01" performance.beta_start="2024-12-01" performance.start="2025-01-01" performance.end="2025-02-01" performance.entry_threshold=3.0 performance.exit_threshold=0.0 performance.stop_loss=2.0 performance.z_score_window=120,216 pair_selection.top_n=20;
```

### MICRO

```bash
poetry run python runners/run_backtest.py -m hydra/launcher=joblib hydra.launcher.n_jobs=-1 clean_single_backtests=false generate_plots=false save_for_training=false rl_model_folder="recurrent_ppo_*_*_*_*_seed*" performance.use_rl=true  performance.autonomous_agent=false performance.iterations=12 performance.beta_hedge=rolling performance.sl_lock=true performance.time_decay_sl=true pair_selection.top_n=20 pair_selection.start="2024-11-01" pair_selection.end="2025-01-01" performance.beta_start="2024-12-01" performance.start="2025-01-01" performance.end="2025-02-01" performance.entry_threshold=2.90,3.10 performance.exit_threshold=0.0 performance.stop_loss=2.0 performance.z_score_window=168;
poetry run python runners/run_backtest.py -m hydra/launcher=joblib hydra.launcher.n_jobs=-1 clean_single_backtests=false generate_plots=false save_for_training=false rl_model_folder="recurrent_ppo_*_*_*_*_seed*" performance.use_rl=true  performance.autonomous_agent=false performance.iterations=12 performance.beta_hedge=rolling performance.sl_lock=true performance.time_decay_sl=true pair_selection.top_n=20 pair_selection.start="2024-11-01" pair_selection.end="2025-01-01" performance.beta_start="2024-12-01" performance.start="2025-01-01" performance.end="2025-02-01" performance.entry_threshold=3.0 performance.exit_threshold=-0.1,0.1 performance.stop_loss=2.0 performance.z_score_window=168;
poetry run python runners/run_backtest.py -m hydra/launcher=joblib hydra.launcher.n_jobs=-1 clean_single_backtests=false generate_plots=false save_for_training=false rl_model_folder="recurrent_ppo_*_*_*_*_seed*" performance.use_rl=true  performance.autonomous_agent=false performance.iterations=12 performance.beta_hedge=rolling performance.sl_lock=true performance.time_decay_sl=true pair_selection.top_n=20 pair_selection.start="2024-11-01" pair_selection.end="2025-01-01" performance.beta_start="2024-12-01" performance.start="2025-01-01" performance.end="2025-02-01" performance.entry_threshold=3.0 performance.exit_threshold=0.0 performance.stop_loss=1.9,2.1 performance.z_score_window=168;
poetry run python runners/run_backtest.py -m hydra/launcher=joblib hydra.launcher.n_jobs=-1 clean_single_backtests=false generate_plots=false save_for_training=false rl_model_folder="recurrent_ppo_*_*_*_*_seed*" performance.use_rl=true  performance.autonomous_agent=false performance.iterations=12 performance.beta_hedge=rolling performance.sl_lock=true performance.time_decay_sl=true pair_selection.top_n=20 pair_selection.start="2024-11-01" pair_selection.end="2025-01-01" performance.beta_start="2024-12-01" performance.start="2025-01-01" performance.end="2025-02-01" performance.entry_threshold=3.0 performance.exit_threshold=0.0 performance.stop_loss=2.0 performance.z_score_window=168 pair_selection.top_n=18,22;
poetry run python runners/run_backtest.py -m hydra/launcher=joblib hydra.launcher.n_jobs=-1 clean_single_backtests=false generate_plots=false save_for_training=false rl_model_folder="recurrent_ppo_*_*_*_*_seed*" performance.use_rl=true  performance.autonomous_agent=false performance.iterations=12 performance.beta_hedge=rolling performance.sl_lock=true performance.time_decay_sl=true pair_selection.top_n=20 pair_selection.start="2024-11-01" pair_selection.end="2025-01-01" performance.beta_start="2024-12-01" performance.start="2025-01-01" performance.end="2025-02-01" performance.entry_threshold=3.0 performance.exit_threshold=0.0 performance.stop_loss=2.0 performance.z_score_window=162,174 pair_selection.top_n=20;
```

### ASSUMPTIONS VERIFICATION

```bash
poetry run python runners/run_backtest.py -m hydra/launcher=joblib hydra.launcher.n_jobs=-1 clean_single_backtests=false generate_plots=false save_for_training=false rl_model_folder="recurrent_ppo_*_*_*_*_seed*" performance.use_rl=true  performance.autonomous_agent=false performance.iterations=12 performance.beta_hedge=rolling performance.sl_lock=true performance.time_decay_sl=true pair_selection.top_n=20 pair_selection.start="2024-11-01" pair_selection.end="2025-01-01" performance.beta_start="2024-12-01" performance.start="2025-01-01" performance.end="2025-02-01" performance.entry_threshold=3.0 performance.exit_threshold=0.0 performance.stop_loss=2.0 performance.z_score_window=168 market.fee_rate=0,0.001;
poetry run python runners/run_backtest.py -m hydra/launcher=joblib hydra.launcher.n_jobs=-1 clean_single_backtests=false generate_plots=false save_for_training=false rl_model_folder="recurrent_ppo_*_*_*_*_seed*" performance.use_rl=true  performance.autonomous_agent=false performance.iterations=12 performance.beta_hedge=rolling performance.sl_lock=true performance.time_decay_sl=true pair_selection.top_n=20 pair_selection.start="2024-11-01" pair_selection.end="2025-01-01" performance.beta_start="2024-12-01" performance.start="2025-01-01" performance.end="2025-02-01" performance.entry_threshold=3.0 performance.exit_threshold=0.0 performance.stop_loss=2.0 performance.z_score_window=168 performance.beta_hedge=no_hedge;
poetry run python runners/run_backtest.py -m hydra/launcher=joblib hydra.launcher.n_jobs=-1 clean_single_backtests=false generate_plots=false save_for_training=false rl_model_folder="recurrent_ppo_*_*_*_*_seed*" performance.use_rl=true  performance.autonomous_agent=false performance.iterations=12 performance.beta_hedge=rolling performance.sl_lock=true performance.time_decay_sl=true pair_selection.top_n=20 pair_selection.start="2024-11-01" pair_selection.end="2025-01-01" performance.beta_start="2024-12-01" performance.start="2025-01-01" performance.end="2025-02-01" performance.entry_threshold=3.0 performance.exit_threshold=0.0 performance.stop_loss=2.0 performance.z_score_window=168 performance.sl_lock=false;
poetry run python runners/run_backtest.py -m hydra/launcher=joblib hydra.launcher.n_jobs=-1 clean_single_backtests=false generate_plots=false save_for_training=false rl_model_folder="recurrent_ppo_*_*_*_*_seed*" performance.use_rl=true  performance.autonomous_agent=false performance.iterations=12 performance.beta_hedge=rolling performance.sl_lock=true performance.time_decay_sl=true pair_selection.top_n=20 pair_selection.start="2024-11-01" pair_selection.end="2025-01-01" performance.beta_start="2024-12-01" performance.start="2025-01-01" performance.end="2025-02-01" performance.entry_threshold=3.0 performance.exit_threshold=0.0 performance.stop_loss=2.0 performance.z_score_window=168 performance.time_decay_sl=false;
poetry run python runners/run_backtest.py -m hydra/launcher=joblib hydra.launcher.n_jobs=-1 clean_single_backtests=false generate_plots=false save_for_training=false rl_model_folder="recurrent_ppo_*_*_*_*_seed*" performance.use_rl=true  performance.autonomous_agent=true performance.iterations=12 performance.beta_hedge=rolling performance.sl_lock=true performance.time_decay_sl=true pair_selection.top_n=20 pair_selection.start="2024-11-01" pair_selection.end="2025-01-01" performance.beta_start="2024-12-01" performance.start="2025-01-01" performance.end="2025-02-01" performance.entry_threshold=3.0 performance.exit_threshold=0.0 performance.stop_loss=2.0 performance.z_score_window=168 performance.time_decay_sl=true;
```

Then you can use `rl_sensitivity_analysis.py` helper to generate PDF plots and table for the analysis.
