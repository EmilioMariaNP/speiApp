import pandas as pd
import spei as si
import scipy.stats as sps
import numpy as np


def calculate_spei_indices(df, spei_list, wb_col='water_balance', date_col='date',
                           start_year=None, end_year=None, dist=sps.fisk):
    """
    Calculates SPEI indices with customizable distributions and missing value handling.
    Fits the distribution on a calibration period and applies it to the whole series.

    Args:
        df (pd.DataFrame): Input data.
        spei_list (list): List of scales (e.g., [3, 12] or ['spei3', 'spei12']).
        wb_col (str): Column for water balance (P - PET). Default 'water_balance'.
        date_col (str): Column for time data. Default 'date'.
        start_year (int): Start year for calibration period.
        end_year (int): End year for calibration period.
        dist (scipy.stats.rv_continuous): Distribution from scipy.stats.
            Available distributions:
            - sps.fisk (Log-Logistic/Fisk) -> Default for SPEI.
            - sps.gamma (Gamma) -> Standard for SPI.
            - sps.genextreme (GEV) -> For extreme value analysis.
            - sps.norm (Normal) -> For standard Z-score type indices.
            - sps.pearson3 (Pearson Type III) -> Common in hydrology.
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).set_index(date_col)

    # Handle missing values
    clean_wb = df[wb_col].interpolate(method='linear', limit=2)

    results = df[[wb_col]].copy()

    for scale in spei_list:
        s_val = int(''.join(filter(str.isdigit, str(scale))))

        if start_year and end_year:
            # 1. Accumulate water balance over the timescale (Standard SPEI procedure)
            rolling_wb = clean_wb.rolling(s_val).sum()

            # 2. Extract the calibration period and drop NaNs
            fit_series = rolling_wb.loc[str(start_year):str(end_year)].dropna()

            # 3. Fit the chosen scipy distribution to the calibration data ONLY
            # dist.fit returns a tuple of parameters (shape, loc, scale)
            params = dist.fit(fit_series)

            # 4. Transform the ENTIRE series using those fitted parameters
            # First, calculate the Cumulative Distribution Function (CDF)
            valid_wb = rolling_wb.dropna()
            cdf_values = dist.cdf(valid_wb, *params)

            # Clip CDF to avoid exact 0s and 1s (which cause infinity when converting to Z-scores)
            cdf_values = np.clip(cdf_values, 1e-6, 1 - 1e-6)

            # Convert CDF to standard normal Z-scores (This is the SPEI value!)
            spei_array = sps.norm.ppf(cdf_values)

            # Safely map the calculated array back to a Pandas Series with the correct time index
            spei_series = pd.Series(index=rolling_wb.index, dtype=float)
            spei_series.loc[valid_wb.index] = spei_array

        else:
            # Fit and transform on entire series using the default package behavior
            spei_values = si.spei(clean_wb, timescale=s_val, dist=dist)
            spei_series = pd.Series(spei_values, index=spei_values.index)

        # Clean infinite values and assign to the results dataframe
        results[f'spei{s_val}'] = spei_series.replace([np.inf, -np.inf], np.nan)

    return results.reset_index()