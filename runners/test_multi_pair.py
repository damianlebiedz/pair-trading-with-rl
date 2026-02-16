import logging
import os
import hydra
import pandas as pd
from omegaconf import OmegaConf, DictConfig

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
    merge_multi_period_results, setup_rl_run_environment,
)
from runners.core.utils import generate_date_lists, load_model

logger = logging.getLogger(__name__)

# =======================================================
best_params = {
    "fixed_window": None,
    "entry_threshold": 2.0,
    "exit_threshold": 0.0,
    "stop_loss": 2,
}
# =======================================================


@hydra.main(version_base=None, config_path="../conf", config_name="base")
def test_multi_pair(cfg: DictConfig):
    root = setup_run_environment(__file__)

    rl_output_dir = None
    if cfg.performance.rl:
        rl_output_dir = setup_rl_run_environment(__file__)

    config = {
        "pair_selection_start": cfg.pair_selection.start,
        "pair_selection_end": cfg.pair_selection.end,
        "test_win_start": cfg.performance.test.win_start,
        "test_start": cfg.performance.test.start,
        "test_end": cfg.performance.test.end,
    }

    number_of_iterations = cfg.performance.iterations
    lists = generate_date_lists(config, number_of_iterations)

    logger.info(f"Saving results to: {root}")
    logger.info("CONFIG:\n%s", OmegaConf.to_yaml(cfg))

    for i in range(number_of_iterations):
        output_dir = os.path.join(root, f"{i+1}")
        if number_of_iterations == 1:
            output_dir = root

        logger.info(f"--- Running Iteration {i+1} ---")

        ps_df = execute_pair_selection(
            tickers=cfg.tickers,
            ps_start=lists["pair_selection_start_list"][i],
            ps_end=lists["pair_selection_end_list"][i],
            test_win_start=lists["test_win_start_list"][i],
            interval=cfg.market.interval,
            top_n_factor=cfg.pair_selection.top_n_factor,
            output_dir=output_dir,
            coint_type=cfg.pair_selection.coint_type,
            beta_method=cfg.performance.beta_method,
            valid_window=(cfg.performance.window_min, cfg.performance.window_max),
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
        if cfg.performance.rl:
            model_path = os.path.join(rl_output_dir, "models")
            try:
                model = load_model(path=model_path)
                agent = RLAgentAdapter(model=model, training_mode=False)
                logger.info("RL Agent loaded successfully and shared across pairs.")
            except Exception as e:
                logger.error(f"Failed to load RL model: {e}")

        for pair_name in selected_pairs_names:
            ticker_x, ticker_y = pair_name.split("-")

            bt = Strategy(
                ticker_x=ticker_x,
                ticker_y=ticker_y,
                start=lists["test_win_start_list"][i],
                end=lists["test_end_list"][i],
                interval=cfg.market.interval,
                fee_rate=cfg.market.fee_rate,
                initial_cash=cfg.market.initial_cash / cfg.pair_selection.top_n_factor,
                risk_free_rate_annual=cfg.market.risk_free_rate_annual,
                min_trades_per_pair=cfg.performance.optimization.min_trades_per_pair,
                beta_hedge=cfg.performance.beta_hedge,
                beta_method=cfg.performance.beta_method,
                window_method=cfg.performance.window_method,
                delayed_entry=cfg.performance.delayed_entry,
                sl_lock=cfg.performance.sl_lock,
                time_decay_sl=(
                    cfg.performance.time_decay_start,
                    cfg.performance.time_decay_end,
                ),
                valid_window=(cfg.performance.window_min, cfg.performance.window_max),
                vol_window=cfg.performance.vol_window,
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
                cfg=cfg,
                bt=bt,
                best_params=best_params,
                ticker_x=ticker_x,
                ticker_y=ticker_y,
                output_dir=output_dir,
                win_test_start=lists["test_win_start_list"][i],
                test_start=lists["test_start_list"][i],
                test_end=lists["test_end_list"][i],
                subdir="test",
            )

            test_results.append(result_test)

        logger.info("--- Merging Multi-Pair Results ---")

        merge_multi_pair_results(
            cfg=cfg,
            output_dir=output_dir,
            results=test_results,
            initial_cash=cfg.market.initial_cash,
            risk_free_rate_annual=cfg.market.risk_free_rate_annual,
            test_start=lists["test_start_list"][i],
            test_end=lists["test_end_list"][i],
        )

    merge_multi_period_results(
        cfg=cfg,
        output_dir=root,
        ticker_x="multi",
        ticker_y="pair",
        initial_cash=cfg.market.initial_cash,
        risk_free_rate_annual=cfg.market.risk_free_rate_annual,
    )

    logger.info(f"Results merged and saved in {root}.")


if __name__ == "__main__":
    test_multi_pair()
