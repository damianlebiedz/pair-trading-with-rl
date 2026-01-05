import logging
import hydra
from omegaconf import DictConfig, OmegaConf
from skopt.space import Real

from modules.data_services.data_utils import save_strategy_result, load_btc_benchmark
from modules.performance.strategy import Strategy
from modules.visualization.plots import plot_positions, plot_pnl, plot_zscore

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def opt(cfg: DictConfig) -> None:
    logger.info("CONFIG:\n%s", OmegaConf.to_yaml(cfg))

    interval = cfg.market.interval
    fee_rate = cfg.market.fee_rate
    initial_cash = cfg.market.initial_cash
    risk_free_rate = cfg.market.risk_free_rate_annual

    window = cfg.performance.window
    source = cfg.performance.source
    beta_hedge = cfg.performance.beta_hedge

    pair_selection_start = cfg.pair_selection.start

    opt_start = cfg.performance.optimization.start
    opt_end = cfg.performance.optimization.end

    ticker_x = "BNBUSDT"
    ticker_y = "UNIUSDT"

    logger.info(f"{ticker_x}-{ticker_y}")

    bt = Strategy(
        ticker_x=ticker_x,
        ticker_y=ticker_y,
        start=pair_selection_start,
        end=opt_end,
        interval=interval,
        fee_rate=fee_rate,
        initial_cash=initial_cash,
        risk_free_rate_annual=risk_free_rate,
        window=window,
        source=source,
        beta_hedge=beta_hedge,
    )

    static_params = {
        # "stop_loss": 2
    }
    param_space = [
        Real(0.5, 2, name="window_factor"),
        Real(1.01, 4.00, name="entry_threshold"),
        Real(0.0, 1.00, name="exit_threshold"),
        Real(1.01, 2.00, name="stop_loss"),
    ]

    metric = ("objective", "net")

    best_params, best_score = bt.run_optimization(
        static_params=static_params,
        param_space=param_space,
        metric=metric,
        opt_start=opt_start,
        opt_end=opt_end,
        pair_selection_start=pair_selection_start,
        n_iter=cfg.performance.optimization.n_iter,
        random_state=cfg.performance.optimization.random_state,
        replicates=cfg.performance.optimization.replicates,
        penalty_bad=cfg.performance.optimization.penalty_bad,
    )

    log = (best_params, best_score)
    logger.info(log)

    window_factor = best_params["window_factor"]
    entry_threshold = best_params["entry_threshold"]
    exit_threshold = best_params["exit_threshold"]
    stop_loss = best_params["stop_loss"]

    result = bt.run_strategy(
        window_factor=window_factor,
        entry_threshold=entry_threshold,
        exit_threshold=exit_threshold,
        stop_loss=stop_loss,
        test_start=opt_start,
        test_end=opt_end,
        pair_selection_start=pair_selection_start,
    )

    parquet_file_name = f"opt_{ticker_x}_{ticker_y}"
    save_strategy_result(result=result, file_name=parquet_file_name, overwrite=cfg.overwrite)

    plot_positions(result, directory="opt", save=True, overwrite=cfg.overwrite)
    btc_data = load_btc_benchmark(
        test_start=opt_start,
        test_end=opt_end,
        interval=interval,
    )
    plot_pnl(result, btc_data, directory="opt", save=True, overwrite=cfg.overwrite)
    plot_zscore(result, directory="opt", save=True, overwrite=cfg.overwrite)


if __name__ == "__main__":
    opt()
