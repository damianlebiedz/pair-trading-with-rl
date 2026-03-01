import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from modules.core.enums import CointType, Source, WindowMethod, BetaHedge
from modules.core.indicators import (
    calculate_beta,
    calculate_half_life_window,
    calculate_hurst,
)
from modules.core.statistical_tests import (
    johansen_cointegration,
    engle_granger_cointegration,
)
from modules.data_services.data_loaders import load_data

logger = logging.getLogger(__name__)


class PairSelector:
    def __init__(
        self,
        coint_type: CointType,
        valid_window: tuple[int, int],
        source: Source = Source.LOG,
    ):
        """
        Initializes the PairSelector module.

        This module is responsible for identifying tradable pairs using a two-stage process:
        1. **Ranking**: Based on a Composite Score (Cointegration Strength + Correlation Quality).
        2. **Validation**: Based on Mean Reversion characteristics (Hurst Exponent) and Beta stability.

        Args:
            coint_type: The statistical test used for the cointegration component of the score.
                - 'eg': Engle-Granger two-step method.
                - 'johansen': Johansen test.
            valid_window (tuple(int, int)): Min and max Z-Score window.
            source: Data transformation applied before analysis (default: 'log').

        Raises:
            ValueError: If `coint_type` or `beta_method` are not supported.
        """
        self.coint_type = coint_type
        self.valid_window = valid_window
        self.source = source

    def select_pairs(
        self,
        tickers: list[str],
        ps_start: str,
        ps_end: str,
        test_win_start: str,
        interval: str,
        top_n: int,
        beta_hedge: BetaHedge,
        window_method: WindowMethod,
        fixed_window: float,
        valid_window: tuple[int, int],
    ) -> pd.DataFrame:
        """
        Executes the Pair Selection pipeline using a Composite Score (Ranking & Validation) approach.

        The process prioritizes pairs that exhibit BOTH strong long-term equilibrium (Cointegration)
        and strong short-term linear dependency (Correlation/$R^2$).

        The process consists of two main phases:
        1. **Scoring & Ranking**: Calculates a weighted score for all pairs on historical data (`ps_start` to `ps_end`).
           Score = 0.5 * Norm(Cointegration) + 0.5 * R_Squared.
        2. **Validation**: Verifies if the top candidates maintain mean-reverting properties
           on the calibration data (`test_start` to `test_end`) using Beta and Hurst.

        Algorithm Stages:
        -----------------
        1. **Data Loading (Selection)**: Loads price data for the `ps_start` - `ps_end` period.
        2. **Composite Scoring**:
           - Runs Cointegration Test (Johansen/EG) -> Normalizes result to 0-1 scale.
           - Calculates Correlation ($R^2$) -> Already 0-1 scale.
           - Computes `Score = (0.5 * Norm_Coint) + (0.5 * R_Squared)`.
           - Sorts pairs by `Score` in descending order. This filters out pairs with high cointegration
             but weak hedging capability (low beta/correlation).
        3. **Data Loading (Validation)**: Loads price data for the `test_start` - `test_end` period.
           This serves as the 'Calibration' period for trading parameters.
        4. **Iterative Validation**:
           Iterates through the top-scored candidates and checks on Validation Data:
           - **Beta Check**: Rejects if Beta <= 0.
           - **Hurst Check**: Calculates Hurst Exponent on the spread formed by the current Beta.
             Rejects if Hurst > 0.5 (indicating trending/random walk behavior).
        5. **Final Selection**: Picks the first `top_n` pairs that pass all validation filters.

        Returns:
            pd.DataFrame: DataFrame containing the selected `top_n` pairs with their
            validation metrics (Beta, Hurst, Window). Returns empty DataFrame if no pairs found.
        """

        logger.debug(f"Loading data for Pair Selection: {ps_start} - {ps_end}")
        df_ps = load_data(
            tickers=tickers, start=ps_start, end=ps_end, interval=interval
        )

        candidates = self._run_scoring_ranking(df_ps)

        if candidates.empty:
            logger.warning("No candidates found after cointegration tests.")
            return pd.DataFrame()

        logger.debug(
            f"Pre-ranked {len(candidates)} pairs. Validating with Hurst, Beta & Window..."
        )

        df_val = load_data(
            tickers=tickers, start=test_win_start, end=ps_end, interval=interval
        )

        for col in df_val.columns and self.source == Source.LOG:
            if df_val[col].dtype in ["float64", "float32"]:
                df_val[f"{col}_{Source.LOG}"] = np.log(df_val[col])

        validated_pairs = []

        for idx, row in candidates.iterrows():
            pair = row["pair"]
            t_x, t_y = pair.split("-")

            try:
                source_x_col = f"{t_x}_{self.source}"
                source_y_col = f"{t_y}_{self.source}"

                X_vals = df_val[source_x_col].values
                Y_vals = df_val[source_y_col].values

                if beta_hedge != BetaHedge.NO_HEDGE:
                    beta = calculate_beta(X_slice=X_vals, Y_slice=Y_vals)
                else:
                    beta = 1

                if beta <= 0:
                    logger.debug(f"Pair {pair} rejected. Beta {beta:.3f} <= 0")
                    continue

                hurst = calculate_hurst(
                    X_slice=X_vals,
                    Y_slice=Y_vals,
                    beta=beta,
                )

                if hurst > 0.5:
                    logger.debug(f"Pair {pair} rejected. Hurst {hurst:.3f} > 0.5")
                    continue

                X_log_vals = df_val[f"{t_x}_{self.source}"].values
                Y_log_vals = df_val[f"{t_y}_{self.source}"].values

                if window_method == "fixed":
                    win = (
                        None
                        if valid_window[0] > fixed_window
                        or valid_window[1] < fixed_window
                        else fixed_window
                    )
                else:
                    win = calculate_half_life_window(
                        X_slice=X_log_vals,
                        Y_slice=Y_log_vals,
                        beta=beta,
                        valid_window=self.valid_window,
                        window_param=fixed_window,
                    )

                if win is None:
                    logger.debug(f"Pair {pair} rejected. Window = None")
                    continue

                res_row = row.to_dict()
                res_row.update(
                    {
                        "validation_beta": beta,
                        "validation_hurst": hurst,
                        "validation_window": win,
                    }
                )
                validated_pairs.append(res_row)

                if len(validated_pairs) >= top_n:
                    break

            except Exception as e:
                logger.error(f"Error validating {pair}: {e}")
                continue

        final_df = pd.DataFrame(validated_pairs)

        if not final_df.empty:
            logger.debug(f"Selected Top {len(final_df)} pairs.")
            return final_df.reset_index(drop=True)
        else:
            return pd.DataFrame()

    def _run_scoring_ranking(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Runs cointegration tests AND correlation analysis to compute a Composite Score.

        Score Logic:
        Score = 0.5 * Normalized(Coint_Strength) + 0.5 * R_Squared

        Returns:
            DataFrame sorted by 'score' (descending).
        """
        if self.coint_type == "johansen":
            res = johansen_cointegration(df)
            scaler = MinMaxScaler()
            res["norm_coint"] = scaler.fit_transform(res[["trace_stat"]])
        else:
            res = engle_granger_cointegration(df)
            res["norm_coint"] = 1 - res["p_value"]

        df_corr = df.copy()
        if self.source == Source.LOG:
            for col in df_corr.columns:
                if df_corr[col].min() > 0:
                    df_corr[col] = np.log(df_corr[col] + 1e-8)

        r2_values = []
        for pair in res["pair"]:
            t_x, t_y = pair.split("-")
            try:
                x_vals = df_corr[t_x].values
                y_vals = df_corr[t_y].values
                corr = np.corrcoef(x_vals, y_vals)[0, 1]
                r2 = corr**2
            except Exception as e:
                logger.error(f"Error calculating correlation {pair}: {e}")
                r2 = 0.0
            r2_values.append(r2)

        res["r_squared"] = r2_values

        w_coint = 0.5
        w_r2 = 0.5

        res["score"] = (w_coint * res["norm_coint"]) + (w_r2 * res["r_squared"])

        return res.sort_values(by="score", ascending=False)
