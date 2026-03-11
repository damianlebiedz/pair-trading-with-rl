import logging
import numpy as np
import pandas as pd

from modules.core.enums import Source
from modules.core.indicators import (
    calculate_beta,
    calculate_hurst,
)
from modules.core.statistical_tests import engle_granger_cointegration
from modules.data_services.data_loaders import load_data

logger = logging.getLogger(__name__)


class PairSelector:
    def __init__(
        self,
        source: Source = Source.LOG.value,
    ):
        """
        Initializes the PairSelector module.

        This module is responsible for identifying tradable pairs using a two-stage process:
        1. **Ranking**: Based on a Composite Score (Cointegration Strength + Correlation Quality).
        2. **Validation**: Based on Mean Reversion characteristics (Hurst Exponent) and Beta stability.

        Args:
            source: Data transformation applied before analysis (default: 'log').

        Raises:
            ValueError: If `coint_type` or `beta_method` are not supported.
        """
        self.source = source

    def select_pairs(
        self,
        tickers: list[str],
        ps_start: str,
        ps_end: str,
        interval: str,
    ) -> pd.DataFrame:
        """
        Executes the Pair Selection pipeline using a Composite Score with Penalty logic.

        The process prioritizes pairs that exhibit BOTH strong long-term equilibrium (Cointegration)
        and strong short-term linear dependency (Correlation/R-squared). Instead of strict rejection
        during validation, pairs that do not meet mean-reversion criteria are penalized.

        The process consists of two main phases:
        1. Scoring & Ranking: Calculates an initial weighted score for all pairs based on
           historical data (ps_start to ps_end).
        2. Validation & Penalty: Verifies if candidates maintain mean-reverting properties
           on calibration data using Beta and Hurst. If pair fails these checks, its score
           is reset to 0.0. Note: Validation metrics are calculated using data ending at
           exactly 00:00:00 on the ps_end date.

        Score Calculation Logic:
        ------------------------
        The final score is determined by the initial statistical quality and validation results:
        - If Hurst is less than or equal to 0.5 AND Beta is greater than 0:
          Score = (0.5 * Normalized Cointegration) + (0.5 * R-squared)
        - If Hurst is greater than 0.5 OR Beta is less than or equal to 0:
          Score = 0.0

        Algorithm Stages:
        -----------------
        1. Data Loading (Selection): Loads price data for the ps_start - ps_end period.
        2. Composite Scoring:
           - Runs Cointegration Test (EG) and normalizes the result to a 0-1 scale.
           - Calculates Correlation (R-squared).
           - Computes initial Score = (0.5 * Norm_Coint) + (0.5 * R_Squared).
           - Sorts pairs by initial Score in descending order.
        3. Data Loading (Validation): Loads price data for the beta_test_start - ps_end period.
           Due to the SoC-EoO model, this calibration data concludes at exactly 00:00:00
           of the ps_end date.
        4. Validation & Penalty:
           Iterates through candidates and checks metrics on Validation Data:
           - Beta Check: If Beta is 0 or less, the pair's score is reset to 0.0.
           - Hurst Check: If Hurst is greater than 0.5 (indicating a trend), the score is reset to 0.0.
        5. Final Selection: Returns all processed pairs, re-sorted by their final Score,
           allowing the strategy to select the top_n valid candidates.

        Returns:
            pd.DataFrame: DataFrame containing all pairs with their validation metrics
            (Beta, Hurst) and the final Score.
        """
        logger.debug(f"Loading data for Pair Selection: {ps_start} - {ps_end}")
        df_ps = load_data(
            tickers=tickers, start=ps_start, end=ps_end, interval=interval
        )
        source_df_ps = np.log(df_ps) if self.source == Source.LOG.value else df_ps
        candidates = self._run_scoring_ranking(source_df_ps)

        if candidates.empty:
            logger.warning("No candidates found after cointegration tests.")
            return pd.DataFrame()

        logger.debug(
            f"Pre-ranked {len(candidates)} pairs. Validating with Hurst, Beta & Window..."
        )

        validated_pairs = []

        for idx, row in candidates.iterrows():
            pair = row["pair"]
            t_x, t_y = pair.split("-")

            try:
                source_x_col = f"{t_x}_{self.source}"
                source_y_col = f"{t_y}_{self.source}"

                X_vals_full = df_ps[source_x_col].values
                Y_vals_full = df_ps[source_y_col].values

                res_row = row.to_dict()

                beta = calculate_beta(X_slice=X_vals_full, Y_slice=Y_vals_full)

                if beta <= 0:
                    logger.debug(
                        f"Pair {pair} penalized. Beta {beta:.3f} <= 0. Score reset to 0."
                    )
                    res_row["score"] = 0.0

                hurst = calculate_hurst(
                    X_slice=X_vals_full,
                    Y_slice=Y_vals_full,
                    beta=beta,
                )

                if hurst > 0.5:
                    logger.debug(
                        f"Pair {pair} penalized. Beta {hurst:.3f} > 0.5. Score reset to 0."
                    )
                    res_row["score"] = 0.0

                res_row = row.to_dict()
                res_row.update(
                    {
                        "validation_beta": beta,
                        "validation_hurst": hurst,
                    }
                )
                validated_pairs.append(res_row)

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

        The score combines long-term equilibrium (cointegration) with short-term linear
        dependency (R-squared) to rank potential pairs.

        Cointegration Normalization Logic:
        - Johansen: Uses percentile ranking (`rank(pct=True)`) on the trace statistic.
          This ensures a uniform distribution between 0.0 and 1.0 and makes the scoring
          robust against extreme outliers that could distort distance-based scalers.
        - Engle-Granger: Uses `1 - p_value`. This inverts the p-value scale so that
          higher values (approaching 1) represent stronger statistical significance
          of the cointegration.

        Score Logic:
        Score = 0.5 * Normalized(Coint_Strength) + 0.5 * R_Squared

        Args:
            df (pd.DataFrame): Historical price data for the assets.

        Returns:
            pd.DataFrame: DataFrame containing test statistics, R-squared values,
            and the final composite 'score', sorted by 'score' in descending order.
        """
        res = engle_granger_cointegration(df)
        res["norm_coint"] = 1 - res["p_value"]

        df_corr = df.copy()
        if self.source == Source.LOG.value:
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
