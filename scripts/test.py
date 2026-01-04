import logging
import hydra
from omegaconf import DictConfig, OmegaConf

from modules.data_services.data_utils import load_btc_benchmark, save_strategy_result
from modules.performance.strategy import Strategy
from modules.visualization.plots import plot_positions, plot_zscore, plot_pnl

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

    test_beta_calculation_start = cfg.performance.test.beta_start
    test_start = cfg.performance.test.start
    test_end = cfg.performance.test.end

    bt = Strategy(
        ticker_x=ticker_x,
        ticker_y=ticker_y,
        start=test_beta_calculation_start,
        end=test_end,
        interval=interval,
        fee_rate=fee_rate,
        initial_cash=initial_cash,
        risk_free_rate_annual=risk_free_rate,
        source=source,
        beta_hedge=beta_hedge,
    )

    entry_threshold = 2.5
    exit_threshold = 0.8
    stop_loss = 1.2
    rolling_window = 50

    result = bt.run_strategy(
        rolling_window=rolling_window,
        entry_threshold=entry_threshold,
        exit_threshold=exit_threshold,
        stop_loss=stop_loss,
        test_start=test_start,
        test_end=test_end,
        beta_calculation_start=test_beta_calculation_start,
    )

    parquet_file_name = f"test_{ticker_x}_{ticker_y}"
    save_strategy_result(result=result, file_name=parquet_file_name)

    plot_positions(result, directory="test", save=True)
    btc_data = load_btc_benchmark(
        test_start=test_start,
        test_end=test_end,
        interval=interval,
    )
    plot_pnl(result, btc_data, directory="test", save=True)
    plot_zscore(result, directory="test", save=True)


if __name__ == "__main__":
    main()
