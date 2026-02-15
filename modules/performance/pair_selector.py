import logging
from typing import Literal
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

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
        coint_type: Literal["eg", "johansen"],
        beta_method: Literal["ols", "johansen", "kalman"],
        valid_window: tuple[int, int],
        source: str = "log",
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
            beta_method: The method used to calculate the Hedge Ratio (Beta) during validation.
                - 'ols': Ordinary Least Squares (static).
                - 'johansen': Vector Error Correction Model (static).
                - 'kalman': Kalman Filter (dynamic state space).
            valid_window: Min and max values of Z-Score window.
            source: Data transformation applied before analysis (default: 'log').

        Raises:
            ValueError: If `coint_type` or `beta_method` are not supported.
        """
        self.coint_type = coint_type
        self.beta_method = beta_method
        self.valid_window = valid_window
        self.source = source

        if coint_type not in ["eg", "johansen"]:
            raise ValueError("'coint_type' must be 'eg' or 'johansen'")

        if beta_method not in ["ols", "johansen", "kalman"]:
            raise ValueError("'beta_method' must be 'ols', 'johansen', or 'kalman'")

        if valid_window[0] > valid_window[1]:
            raise ValueError(f"'valid_window' should be (min, max): {valid_window}")

    def select_pairs(
        self,
        tickers: list[str],
        ps_start: str,
        ps_end: str,
        test_win_start: str,
        interval: str,
        top_n: int,
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

        Args:
            tickers: List of asset tickers to analyze.
            ps_start: Start date for the Scoring/Ranking data.
            ps_end: End date for the Scoring/Ranking data.
            test_win_start: Start date for the Validation/Calibration data.
            interval: Data timeframe (e.g., '1h').
            top_n: Number of pairs to select.

        Returns:
            pd.DataFrame: DataFrame containing the selected `top_n` pairs with their
            validation metrics (Beta, Hurst, Window). Returns empty DataFrame if no pairs found.
        """

        logger.debug(f"Loading data for Pair Selection: {ps_start} - {ps_end}")
        df_ps = load_data(tickers, ps_start, ps_end, interval)

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

        for col in df_val.columns:
            if df_val[col].dtype in ["float64", "float32"]:
                df_val[f"{col}_log"] = np.log(df_val[col] + 1e-8)

        validated_pairs = []

        for idx, row in candidates.iterrows():
            pair = row["pair"]
            t_x, t_y = pair.split("-")

            try:
                source_x_col = f"{t_x}_{self.source}"
                source_y_col = f"{t_y}_{self.source}"

                beta = calculate_beta(
                    x_col=source_x_col,
                    y_col=source_y_col,
                    df=df_val,
                    beta_method=self.beta_method,
                )

                if beta <= 0:
                    continue

                hurst = calculate_hurst(
                    x_col=source_x_col,
                    y_col=source_y_col,
                    beta=beta,
                    df=df_val,
                )

                if hurst > 0.5:
                    logger.debug(f"Pair {pair} rejected. Hurst {hurst:.3f} > 0.5")
                    continue

                win = calculate_half_life_window(
                    x_col=f"{t_x}_log",
                    y_col=f"{t_y}_log",
                    beta=beta,
                    df=df_val,
                    valid_window=self.valid_window,
                )
                if win is None:
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
        if self.source == "log":
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
