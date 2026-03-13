import json
import logging
import os
import shutil
from pathlib import Path
import hydra
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from modules.core.enums import ObsSpaceType
from modules.core.config import Config
from modules.learning.agents import RLAgentAdapter
from modules.performance.models import StrategyResult
from modules.data_services.data_loaders import load_data
from modules.data_services.data_utils import (
    save_strategy_result,
    save_dataframe,
    load_btc_benchmark,
    load_ewp_benchmark,
)
from modules.performance.strategy import Strategy
from runners.core.pipelines import (
    execute_testing,
    setup_run_environment,
    merge_multi_pair_results,
    merge_multi_period_results,
)
from runners.core.utils import (
    generate_date_lists,
    load_model,
    save_hydra_config_snapshot,
    get_first_subdirectory,
)

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../config", config_name="run_backtest")
def run_backtest(cfg: DictConfig):
    root = setup_run_environment(__file__)
    save_hydra_config_snapshot(cfg=cfg, root_dir=root)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))

    training_session_dir = None
    if cfg.save_for_training:
        run_id = os.path.basename(root)
        training_session_dir = os.path.join(project_root, "data", "rl_training", run_id)
        os.makedirs(training_session_dir, exist_ok=True)
        logger.info(
            f"Backtest results will be saved for training in: {training_session_dir}"
        )

    json_path = os.path.join(project_root, "config/schemas/list_of_assets.json")

    if not os.path.exists(json_path):
        logger.error(f"Error: file not found {json_path}")
        return

    with open(json_path, "r", encoding="utf-8") as f:
        universe_data = json.load(f)

    cfg = Config(**OmegaConf.to_container(cfg, resolve=True))

    best_params = {
        "z_score_window": cfg.performance.z_score_window,
        "entry_threshold": cfg.performance.entry_threshold,
        "exit_threshold": cfg.performance.exit_threshold,
        "stop_loss": cfg.performance.stop_loss,
    }

    config = {
        "pair_selection_start": cfg.pair_selection.start,
        "pair_selection_end": cfg.pair_selection.end,
        "beta_test_start": cfg.performance.beta_start,
        "test_start": cfg.performance.start,
        "test_end": cfg.performance.end,
    }

    number_of_iterations = cfg.performance.iterations
    lists = generate_date_lists(config, number_of_iterations)

    base_ps_dir = os.path.join(project_root, "data", "pair_selection")

    missing_files = []
    for i in range(number_of_iterations):
        ps_start = lists["pair_selection_start_list"][i]
        ps_end = lists["pair_selection_end_list"][i]
        month_key = pd.to_datetime(ps_start).strftime("%Y-%m")

        expected_file = os.path.join(
            base_ps_dir, month_key, f"pair_selection_{ps_start}_{ps_end}.parquet"
        )
        if not os.path.exists(expected_file):
            missing_files.append(f"{month_key} ({ps_start})")

    if missing_files:
        raise ValueError(
            f"Missed pair selection for: {', '.join(missing_files)}. Use run_pair_selection.py!"
        )

    logger.info(f"Saving results to: {root}")

    tickers_dict = {}

    should_cleanup = number_of_iterations > 1

    for i in range(number_of_iterations):
        output_dir = os.path.join(root, f"{i + 1}")
        if number_of_iterations == 1:
            output_dir = root

        logger.info(f"--- Running Iteration {i + 1} / {number_of_iterations} ---")

        ps_start = lists["pair_selection_start_list"][i]
        ps_end = lists["pair_selection_end_list"][i]
        beta_start = lists["beta_test_start_list"][i]
        test_start = lists["test_start_list"][i]
        test_end = lists["test_end_list"][i]

        current_selection_date = pd.to_datetime(ps_start)
        month_key = current_selection_date.strftime("%Y-%m")

        if month_key not in universe_data:
            logger.error(
                f"Universe for month {month_key} not found in list_of_assets.json!"
            )
            continue

        current_iteration_tickers = universe_data[month_key]["assets"]
        logger.debug(
            f"Using universe from {month_key}: {len(current_iteration_tickers)} assets."
        )

        tickers_dict[i + 1] = current_iteration_tickers

        ps_file = os.path.join(
            base_ps_dir, month_key, f"pair_selection_{ps_start}_{ps_end}.parquet"
        )
        ps_df = pd.read_parquet(ps_file)
        if not ps_df.empty:
            ps_df = ps_df.sort_values(by="score", ascending=False)
            ps_df = ps_df.head(cfg.pair_selection.top_n)
        logger.info("\n%s", ps_df.to_string())
        selected_pairs_names = ps_df["pair"].tolist()

        if len(selected_pairs_names) > 1:
            should_cleanup = True

        if not selected_pairs_names:
            logger.warning(
                f"Iteration {i + 1}: No pairs selected! Generating flat (cash-only) result for this period."
            )

            ref_ticker = current_iteration_tickers[0]
            ref_data = load_data(
                tickers=[ref_ticker],
                start=test_start,
                end=test_end,
                interval=cfg.market.interval,
            )

            empty_data = pd.DataFrame(index=ref_data.index)
            empty_data["total_pnl"] = 0.0
            empty_data["total_net_pnl"] = 0.0
            empty_data["total_return"] = 0.0
            empty_data["total_net_return"] = 0.0
            empty_data["in_position"] = 0.0

            empty_result = StrategyResult(
                data=empty_data,
                ticker_x="multi",
                ticker_y="pair",
                start=test_start,
                end=test_end,
                interval=cfg.market.interval,
                fee_rate=cfg.market.fee_rate,
                stats=pd.DataFrame(),
                exec_logger=pd.DataFrame(),
            )

            save_strategy_result(
                result=empty_result,
                file_name=f"returns_multi_pair_{empty_result.start}_{empty_result.end}",
                directory=output_dir,
            )

            save_dataframe(
                df=pd.DataFrame(),
                file_name=f"exec_logger_multi_pair_{empty_result.start}_{empty_result.end}",
                directory=output_dir,
            )

            save_dataframe(
                df=pd.DataFrame(),
                file_name=f"stats_multi_pair_{empty_result.start}_{empty_result.end}",
                directory=output_dir,
            )

            continue

        strategies = []
        strategies_map = {}

        rl_model_folder = cfg.rl_model_folder

        agent = None
        if cfg.performance.use_rl:
            if not rl_model_folder:
                models_root = os.path.join(project_root, "data", "rl_models")
                try:
                    rl_model_folder = get_first_subdirectory(models_root)
                    logger.info(
                        f"No model subfolder specified. Auto-selected: {rl_model_folder}"
                    )
                except (FileNotFoundError, ValueError) as e:
                    logger.error(f"Cannot auto-select model: {e}")
                    return

            base_model_path = os.path.join(
                project_root, "data", "rl_models", rl_model_folder
            )

            valid_spaces = ObsSpaceType
            obs_space_type = next(
                (
                    space
                    for space in valid_spaces
                    if f"_{space.value.lower()}_" in rl_model_folder.lower()
                ),
                None,
            )

            if not obs_space_type:
                raise ValueError(
                    f"Error: wrong obs_space_type in model_name: '{rl_model_folder}'. "
                    f"Must be one of: {valid_spaces}"
                )

            model_zip_path = f"{base_model_path}.zip"
            vec_normalize_path = f"{base_model_path}_normalize.pkl"

            if not os.path.exists(model_zip_path):
                raise FileNotFoundError(f"Model file not found: {model_zip_path}")
            if not os.path.exists(vec_normalize_path):
                raise FileNotFoundError(
                    f"Vec-Normalize file not found: {vec_normalize_path}"
                )

            try:
                model, vec_normalize = load_model(
                    model_path=base_model_path,
                    vec_normalize_path=vec_normalize_path,
                    obs_space_type=obs_space_type,
                )
                agent = RLAgentAdapter(
                    model=model,
                    vec_normalize=vec_normalize,
                    training_mode=False,
                    obs_space_type=obs_space_type,
                )
                logger.info("RL Agent loaded successfully.")
            except Exception as e:
                logger.error(f"Failed to load RL model: {e}")

        for pair_name in selected_pairs_names:
            ticker_x, ticker_y = pair_name.split("-")

            bt = Strategy(
                ticker_x=ticker_x,
                ticker_y=ticker_y,
                start=beta_start,
                end=test_end,
                interval=cfg.market.interval,
                fee_rate=cfg.market.fee_rate,
                initial_cash=cfg.market.initial_cash / cfg.pair_selection.top_n,
                risk_free_rate_annual=cfg.market.risk_free_rate_annual,
                beta_hedge=cfg.performance.beta_hedge,
                delayed_entry=cfg.performance.delayed_entry,
                sl_lock=cfg.performance.sl_lock,
                time_decay_sl=cfg.performance.time_decay_sl,
                time_decay_params=(
                    cfg.settings.time_decay_min,
                    cfg.settings.time_decay_max,
                ),
                vol_window=cfg.settings.vol_window,
                freeze_std=cfg.performance.freeze_std,
                agent=agent,
            )

            strategies.append(bt)
            strategies_map[pair_name] = bt

        test_results = []
        logger.info(
            f"--- Testing {len(selected_pairs_names)} Pairs (Test Window: {test_start} to {test_end}) ---"
        )

        btc_data = None
        ewp_data = None
        if cfg.generate_plots:
            btc_data = load_btc_benchmark(
                test_start=test_start,
                test_end=test_end,
                interval=cfg.market.interval,
                fee_rate=cfg.market.fee_rate,
            )
            ewp_data = load_ewp_benchmark(
                tickers=current_iteration_tickers,
                test_start=test_start,
                test_end=test_end,
                interval=cfg.market.interval,
                fee_rate=cfg.market.fee_rate,
            )

        for pair_name in selected_pairs_names:
            ticker_x, ticker_y = pair_name.split("-")
            bt = strategies_map[pair_name]

            if bt.agent is not None:
                bt.agent.reset_agent()
                logger.debug(f"Agent memory reset for pair {pair_name}")

            logger.debug(f"--- Testing pair: {pair_name} ---")

            result_test = execute_testing(
                bt=bt,
                best_params=best_params,
                ticker_x=ticker_x,
                ticker_y=ticker_y,
                output_dir=output_dir,
                beta_test_start=beta_start,
                test_start=test_start,
                test_end=test_end,
                subdir="test",
                plot=cfg.generate_plots,
                btc_data=btc_data,
                ewp_data=ewp_data,
            )

            test_results.append(result_test)

            if cfg.save_for_training:
                file_name = f"returns_{ticker_x}_{ticker_y}_{result_test.start}_{result_test.end}"
                save_strategy_result(
                    result=result_test,
                    file_name=file_name,
                    directory=training_session_dir,
                )

        if len(test_results) > 1:
            merge_multi_pair_results(
                output_dir=output_dir,
                results=test_results,
                initial_cash=cfg.market.initial_cash,
                risk_free_rate_annual=cfg.market.risk_free_rate_annual,
                test_start=test_start,
                test_end=test_end,
                plot=cfg.generate_plots,
                btc_data=btc_data,
                ewp_data=ewp_data,
            )

    if number_of_iterations > 1:
        merge_multi_period_results(
            output_dir=root,
            ticker_x="multi",
            ticker_y="pair",
            initial_cash=cfg.market.initial_cash,
            risk_free_rate_annual=cfg.market.risk_free_rate_annual,
            interval=cfg.market.interval,
            plot=cfg.generate_plots,
            tickers_dict=tickers_dict,
        )

    if cfg.clean_single_backtests and should_cleanup:
        logger.debug("Cleaning up nested 'test' subfolders...")
        deleted_count = 0

        for p in Path(root).rglob("test"):
            if p.is_dir():
                try:
                    shutil.rmtree(p)
                    deleted_count += 1
                    logger.debug(f"Deleted subfolder: {p}")
                except Exception as e:
                    logger.error(f"Error during deleting {p}: {e}")

        logger.info(f"Cleanup finished. Deleted {deleted_count} 'test' subdirs.")

    logger.info(f"Results saved in {root}.")


if __name__ == "__main__":
    run_backtest()
