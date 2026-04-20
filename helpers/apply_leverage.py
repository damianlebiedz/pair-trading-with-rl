"""Script to apply leverage to raw strategy outputs and generate new leveraged datasets."""

import shutil
import sys
import yaml
import pandas as pd
from pathlib import Path

from modules.performance.stats import calculate_stats
from modules.core.enums import Interval
from modules.utils.logger import get_logger

LEVERAGES = [10.0]
BASE_DIR = "RL Sensitivity Analysis/Assumptions Verification/oos"

current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
sys.path.append(str(project_root))

logger = get_logger(__name__)


def generate_leveraged_folders():
    results_dir = project_root / "results"
    base_path = results_dir / BASE_DIR

    if not base_path.exists():
        logger.error(f"Base directory does not exist: {base_path}")
        return

    folders = [
        p for p in base_path.rglob("*") if p.is_dir() and not p.name.startswith(".")
    ]

    for source_dir in folders:
        folder = source_dir.relative_to(results_dir)

        source_hydra = source_dir / ".hydra"
        config_path = source_hydra / "config.yaml"

        if not config_path.exists():
            continue

        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        initial_cash = float(config.get("market", {}).get("initial_cash", 10000))
        risk_free_rate = float(
            config.get("market", {}).get("risk_free_rate_annual", 0.0)
        )

        returns_files = list(source_dir.glob("returns_*.parquet"))
        exec_files = list(source_dir.glob("exec_logger_*.parquet"))
        stats_files = list(source_dir.glob("stats_multi_pair_*.parquet"))

        if not returns_files or not exec_files:
            logger.warning(f"Missing returns/exec files in {folder}, skipping.")
            continue

        df_ret_base = pd.read_parquet(returns_files[0])
        df_exec_base = pd.read_parquet(exec_files[0])

        for lev in LEVERAGES:
            target_dir = results_dir / f"{folder}_lev_{lev:g}"
            target_dir.mkdir(parents=True, exist_ok=True)

            logger.info(f"Processing: {folder} -> {target_dir.name} (Leverage: {lev}x)")

            if source_hydra.exists():
                target_hydra = target_dir / ".hydra"
                if target_hydra.exists():
                    shutil.rmtree(target_hydra)
                shutil.copytree(source_hydra, target_hydra)

            df_ret_lev = df_ret_base.copy()

            rets_base_pct = df_ret_base["equity"].pct_change().fillna(0)
            rets_lev_pct = rets_base_pct * lev

            equity_lev = initial_cash * (1 + rets_lev_pct).cumprod()

            bankruptcy_mask = equity_lev <= 0
            if bankruptcy_mask.any():
                first_bankruptcy_idx = bankruptcy_mask.idxmax()
                equity_lev.loc[first_bankruptcy_idx:] = 0.0

            df_ret_lev["equity"] = equity_lev
            df_ret_lev["total_net_pnl"] = equity_lev - initial_cash

            gross_step_base = (
                df_ret_base["total_pnl"].diff().fillna(df_ret_base["total_pnl"].iloc[0])
            )

            prev_equity_base = df_ret_base["equity"].shift(1).fillna(initial_cash)
            pct_gross_base = (gross_step_base / prev_equity_base).fillna(0)
            prev_equity_lev = equity_lev.shift(1).fillna(initial_cash)

            step_gross_lev = (pct_gross_base * lev) * prev_equity_lev

            df_ret_lev["total_pnl"] = step_gross_lev.cumsum()
            df_ret_lev["total_fees"] = (
                df_ret_lev["total_pnl"] - df_ret_lev["total_net_pnl"]
            )

            df_ret_lev["total_return"] = df_ret_lev["total_pnl"] / initial_cash
            df_ret_lev["total_net_return"] = df_ret_lev["total_net_pnl"] / initial_cash

            df_exec_lev = df_exec_base.copy()

            stats_lev = calculate_stats(
                df_ret_lev, df_exec_lev, initial_cash, Interval.H1, risk_free_rate
            )

            if isinstance(stats_lev, pd.DataFrame):
                df_stats_lev = stats_lev.copy().reset_index()
                first_col_name = df_stats_lev.columns[0]
                df_stats_lev.rename(columns={first_col_name: "metric"}, inplace=True)
            else:
                df_stats_lev = pd.DataFrame(stats_lev).reset_index()
                df_stats_lev.rename(columns={"index": "metric"}, inplace=True)

            ret_name = returns_files[0].name
            exec_name = exec_files[0].name
            stats_name = (
                stats_files[0].name
                if stats_files
                else f"stats_multi_pair_leveraged_{lev:g}.parquet"
            )

            df_ret_lev.to_parquet(target_dir / ret_name)
            df_exec_lev.to_parquet(target_dir / exec_name)
            df_stats_lev.to_parquet(target_dir / stats_name, index=False)

    logger.info("All operations completed successfully.")


if __name__ == "__main__":
    generate_leveraged_folders()
