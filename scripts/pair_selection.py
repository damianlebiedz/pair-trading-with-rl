import logging
import hydra
from omegaconf import DictConfig

from modules.data_services.data_loaders import load_data
from modules.data_services.data_utils import save_dataframe, merge_by_pair
from modules.performance.statistical_tests import engle_granger_cointegration, pearson_correlation, \
    ssd_cumulative_returns

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    interval = cfg.market.interval

    tickers = cfg.pair_selection.tickers
    pair_selection_start = cfg.pair_selection.start
    pair_selection_end = cfg.pair_selection.end

    df = load_data(
        tickers=tickers,
        start=pair_selection_start,
        end=pair_selection_end,
        interval=interval
    )

    ssd_c_returns_df = ssd_cumulative_returns(df)
    corr_log_returns_df = pearson_correlation(df, source="log_returns")
    eg_log_prices_df = engle_granger_cointegration(df, source="log_prices")

    merged_df = merge_by_pair(
        dfs=[ssd_c_returns_df, corr_log_returns_df, eg_log_prices_df],
        keep_cols=[
            ['ssd'],
            ['corr_log_returns'],
            ['eg_p_value']
        ]
    ).sort_values('eg_p_value', ascending=True).reset_index(drop=True)

    save_dataframe(df=merged_df, file_name=f'pair_selection_{pair_selection_start}_{pair_selection_end}')


if __name__ == "__main__":
    main()
