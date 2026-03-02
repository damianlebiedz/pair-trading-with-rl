import logging
import os
import hydra
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from modules.core.enums import ObsSpaceType
from modules.core.config import Config
from modules.learning.agents import RLAgentAdapter
from modules.performance.models import StrategyResult
from modules.data_services.data_loaders import load_data
from modules.data_services.data_utils import save_dataframe, save_strategy_result
from modules.performance.strategy import Strategy
from runners.core.pipelines import (
    execute_pair_selection,
    execute_testing,
    setup_run_environment,
    merge_multi_pair_results,
    merge_multi_period_results,
    setup_rl_run_environment,
)
from runners.core.utils import (
    generate_date_lists,
    load_model,
    save_hydra_config_snapshot,
)

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../config", config_name="run_backtest")
def run_backtest(cfg: DictConfig):
    root = setup_run_environment(__file__)
    save_hydra_config_snapshot(cfg=cfg, root_dir=root)

    cfg = Config(**OmegaConf.to_container(cfg, resolve=True))

    best_params = {
        "z_score_window": cfg.z_score_window,
        "entry_threshold": cfg.entry_threshold,
        "exit_threshold": cfg.exit_threshold,
        "stop_loss": cfg.stop_loss,
    }

    rl_output_dir = None
    if cfg.performance.use_rl:
        rl_output_dir = setup_rl_run_environment(__file__)

    config = {
        "pair_selection_start": cfg.pair_selection.start,
        "pair_selection_end": cfg.pair_selection.end,
        "beta_test_start": cfg.performance.test.beta_start,
        "test_start": cfg.performance.test.start,
        "test_end": cfg.performance.test.end,
    }

    number_of_iterations = cfg.performance.iterations
    lists = generate_date_lists(config, number_of_iterations)

    earliest_date = min(
        pd.to_datetime(lists["pair_selection_start_list"][0]),
        pd.to_datetime(lists["beta_test_start_list"][0]),
        pd.to_datetime(lists["test_start_list"][0]),
    ).strftime("%Y-%m-%d")

    latest_date = max(
        pd.to_datetime(lists["pair_selection_end_list"][-1]),
        pd.to_datetime(lists["test_end_list"][-1]),
    ).strftime("%Y-%m-%d")

    logger.debug(
        f"Pre-validating data availability from {earliest_date} to {latest_date}..."
    )
    try:
        _validation_df = load_data(
            tickers=[cfg.tickers[0]],
            start=earliest_date,
            end=latest_date,
            interval=cfg.market.interval,
        )
        del _validation_df
    except ValueError as e:
        logger.error("Not enough historical data for requested ranges!")
        logger.error(str(e))
        raise SystemExit(
            "Backtest aborted due to missing data. Please fetch more data or adjust dates."
        )

    logger.info(f"Saving results to: {root}")

    tickers = cfg.tickers if cfg.generate_plots else None

    for i in range(number_of_iterations):
        output_dir = os.path.join(root, f"{i+1}")
        if number_of_iterations == 1:
            output_dir = root

        logger.info(f"--- Running Iteration {i+1} ---")

        ps_df = execute_pair_selection(
            tickers=cfg.tickers,
            ps_start=lists["pair_selection_start_list"][i],
            ps_end=lists["pair_selection_end_list"][i],
            beta_test_start=lists["beta_test_start_list"][i],
            interval=cfg.market.interval,
            top_n_factor=cfg.pair_selection.top_n_factor,
            output_dir=output_dir,
            coint_type=cfg.pair_selection.coint_type,
            beta_hedge=cfg.performance.beta_hedge,
        )

        logger.info("\n%s", ps_df.to_string())
        selected_pairs_names = ps_df["pair"].tolist()

        if not selected_pairs_names:
            logger.warning(
                f"Iteration {i + 1}: No pairs selected! Generating flat (cash-only) result for this period."
            )

            ref_ticker = cfg.tickers[0]
            ref_data = load_data(
                tickers=[ref_ticker],
                start=lists["test_start_list"][i],
                end=lists["test_end_list"][i],
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
                start=lists["test_start_list"][i],
                end=lists["test_end_list"][i],
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

        agent = None
        if cfg.performance.use_rl:
            valid_spaces = ObsSpaceType
            obs_space_type = next(
                (
                    space
                    for space in valid_spaces
                    if f"_{space}_" in cfg.performance.model_name
                ),
                None,
            )

            if not obs_space_type:
                raise ValueError(
                    f"Error: wrong obs_space_type in model_name: '{cfg.performance.model_name}'. "
                    f"Must be one of: {valid_spaces}"
                )

            base_model_path = os.path.join(
                rl_output_dir, "models", cfg.performance.model_name
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
                start=lists["beta_test_start_list"][i],
                end=lists["test_end_list"][i],
                interval=cfg.market.interval,
                fee_rate=cfg.market.fee_rate,
                initial_cash=cfg.market.initial_cash / cfg.pair_selection.top_n_factor,
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

        logger.info(f"--- Testing {len(selected_pairs_names)} Pairs ---")

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
                beta_test_start=lists["beta_test_start_list"][i],
                test_start=lists["test_start_list"][i],
                test_end=lists["test_end_list"][i],
                subdir="test",
                interval=cfg.market.interval,
                plot=cfg.generate_plots,
                tickers=tickers,
            )

            test_results.append(result_test)

        if len(test_results) > 1:
            merge_multi_pair_results(
                output_dir=output_dir,
                results=test_results,
                initial_cash=cfg.market.initial_cash,
                risk_free_rate_annual=cfg.market.risk_free_rate_annual,
                test_start=lists["test_start_list"][i],
                test_end=lists["test_end_list"][i],
                interval=cfg.market.interval,
                plot=cfg.generate_plots,
                tickers=tickers,
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
            tickers=tickers,
        )

    logger.info(f"Results saved in {root}.")


if __name__ == "__main__":
    run_backtest()
