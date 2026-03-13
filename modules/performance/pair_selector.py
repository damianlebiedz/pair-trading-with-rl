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
        Executes the Pair Selection pipeline using a Two-Stage "Filter-then-Rank" methodology.

        This process prioritizes pairs that exhibit strong long-term equilibrium (Cointegration)
        and short-term linear dependency (R-squared), while strictly filtering out pairs that
        lack tradable structural characteristics (Mean Reversion and valid Hedge Ratio).

        The process consists of two main phases executed on the same historical data window:
        1. Scoring (Quality Assessment): Calculates an initial composite score for all pairs.
        2. Filtering (Structural Validation): Verifies if candidates are actually tradable
           using the Hurst Exponent and Beta. Fails result in a strict penalty.

        Score Calculation & Penalty Logic:
        ----------------------------------
        Initial Score = (0.5 * Normalized Cointegration) + (0.5 * R-squared)

        A hard penalty (Score = 0.0) is applied if either structural constraint is violated:
        - Hurst > 0.5: Indicates the spread is trending, violating mean-reversion assumptions.
        - Beta <= 0: Indicates assets move inversely or lack a valid linear relationship for hedging.

        Algorithm Stages:
        -----------------
        1. Data Loading: Loads and cleans price data for the ps_start to ps_end period.
        2. Composite Scoring: Evaluates all possible asset combinations, calculates the
           initial Score based on Engle-Granger and R-squared, and ranks them.
        3. Validation & Penalty: Iterates through the ranked candidates. Computes Beta
           and Hurst on the same data window. If constraints fail, the score is zeroed out.
        4. Final Selection: Returns the processed dataframe, allowing the downstream
           strategy to safely select the top_n viable candidates.

        Methodological Notes:
        - Closing Prices: All statistical tests (Cointegration, Correlation, Hurst, Beta)
          are evaluated strictly on historical closing prices to eliminate intraday noise.
        - Logarithmic Transformation: Price series are transformed into natural logarithms
          prior to analysis so that the spread and hedge ratio reflect proportional relationships.

        Args:
            tickers (list[str]): List of asset tickers to evaluate.
            ps_start (str): Start date for the selection data window.
            ps_end (str): End date for the selection data window.
            interval (str): Timeframe interval (e.g., '1h', '1d').

        Returns:
            pd.DataFrame: DataFrame containing all evaluated pairs, their validation metrics
            (validation_beta, validation_hurst), and the final penalized score.
        """
        logger.debug(f"Loading data for Pair Selection: {ps_start} - {ps_end}")
        df_ps = load_data(
            tickers=tickers, start=ps_start, end=ps_end, interval=interval
        )

        assets_with_nans = df_ps.columns[df_ps.isna().any()].tolist()
        if assets_with_nans:
            logger.warning(
                f"Dropping assets with NaNs from Pair Selection: {assets_with_nans}"
            )
            df_ps = df_ps.dropna(axis=1)

        if df_ps.shape[1] < 2:
            logger.warning(
                "Not enough valid assets left to form pairs after dropping NaNs."
            )
            return pd.DataFrame()

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
                X_vals_full = source_df_ps[t_x].values
                Y_vals_full = source_df_ps[t_y].values

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
                        f"Pair {pair} penalized. Hurst {hurst:.3f} > 0.5. Score reset to 0."
                    )
                    res_row["score"] = 0.0

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
