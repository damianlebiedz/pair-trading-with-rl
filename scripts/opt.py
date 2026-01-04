import logging
import hydra
from omegaconf import DictConfig, OmegaConf
from skopt.space import Integer, Real

from modules.data_services.data_utils import save_strategy_result, load_btc_benchmark
from modules.performance.strategy import Strategy
from modules.visualization.plots import plot_positions, plot_pnl, plot_zscore

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    logger.info("CONFIG:\n%s", OmegaConf.to_yaml(cfg))

    interval = cfg.market.interval
    fee_rate = cfg.market.fee_rate
    initial_cash = cfg.market.initial_cash
    risk_free_rate = cfg.market.risk_free_rate_annual

    ticker_x = cfg.performance.ticker_x
    ticker_y = cfg.performance.ticker_y
    source = cfg.performance.source
    beta_hedge = cfg.performance.beta_hedge

    opt_beta_calculation_start = cfg.performance.optimization.beta_start
    opt_start = cfg.performance.optimization.start
    opt_end = cfg.performance.optimization.end

    bt = Strategy(
        ticker_x=ticker_x,
        ticker_y=ticker_y,
        start=opt_beta_calculation_start,
        end=opt_end,
        interval=interval,
        fee_rate=fee_rate,
        initial_cash=initial_cash,
        risk_free_rate_annual=risk_free_rate,
        source=source,
        beta_hedge=beta_hedge,
    )

    static_params = {"stop_loss": 2}
    param_space = [
        Integer(10, 400, name="rolling_window"),  # UWAGA: nie może przekraczać zakresu danych!
        Real(1.01, 4.00, name="entry_threshold"),
        Real(0.0, 1.00, name="exit_threshold"),
        # Real(1.01, 3.00, name="stop_loss"),
    ]

    metric = ("equity_slope_r2", "net")

    best_params, best_score = bt.run_optimization(
        static_params=static_params,
        param_space=param_space,
        metric=metric,
        opt_start=opt_start,
        opt_end=opt_end,
        opt_beta_calculation_start=opt_beta_calculation_start,
        n_iter=cfg.performance.optimization.n_iter,
        random_state=cfg.performance.optimization.random_state,
        replicates=cfg.performance.optimization.replicates,
        penalty_bad=cfg.performance.optimization.penalty_bad,
    )

    log = (best_params, best_score)
    logger.info(log)

    rolling_window = best_params["rolling_window"]
    entry_threshold = best_params["entry_threshold"]
    exit_threshold = best_params["exit_threshold"]
    # stop_loss = best_params["stop_loss"]
    stop_loss = 1.2

    result = bt.run_strategy(
        rolling_window=rolling_window,
        entry_threshold=entry_threshold,
        exit_threshold=exit_threshold,
        stop_loss=stop_loss,
        test_start=opt_start,
        test_end=opt_end,
        beta_calculation_start=opt_beta_calculation_start,
    )

    parquet_file_name = f"opt_{ticker_x}_{ticker_y}"
    save_strategy_result(result=result, file_name=parquet_file_name)

    plot_positions(result, directory="opt", save=True)
    btc_data = load_btc_benchmark(
        test_start=opt_start,
        test_end=opt_end,
        interval=interval,
    )
    plot_pnl(result, btc_data, directory="opt", save=True)
    plot_zscore(result, directory="opt", save=True)


if __name__ == "__main__":
    main()
