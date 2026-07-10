import pandas as pd
import spei as si
import scipy.stats as sps
import numpy as np
from scipy.stats import gamma


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

    # Aggregate duplicate dates, taking the mean of the water balance.
    # This prevents "ValueError: cannot reindex on an axis with duplicate labels"
    if df.index.has_duplicates:
        df = df.groupby(df.index).mean(numeric_only=True)

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

            # Create the SPEI series directly with the calculated values and the correct index
            spei_series = pd.Series(spei_array, index=valid_wb.index)

        else:
            # Fit and transform on entire series using the default package behavior
            spei_values = si.spei(clean_wb, timescale=s_val, dist=dist)
            spei_series = pd.Series(spei_values, index=spei_values.index)

        # Clean infinite values and assign to the results dataframe
        results[f'spei{s_val}'] = spei_series.replace([np.inf, -np.inf], np.nan)

    return results.reset_index()



def calculate_drought_events(df, group_cols, spei_col='spei', threshold=-1.0):
    """
    Identifies drought events based on an SPEI threshold and calculates the expected return
    periods of their duration and intensity using a probability distribution.

    Args:
    - df: pandas DataFrame containing at least 'year', 'month', 'scenario', 'gcm', 'ecoregion', and spei_col
    - spei_col: String, name of the column containing SPEI data
    - threshold: Float, SPEI value below which a month is considered in drought

    Returns:
    - df_out: DataFrame with the new boolean 'drought' column (1 if SPEI < threshold else 0)
    - results_dict: Nested dictionary containing calculated return periods
    """
    # Use a copy to prevent modifying the raw dataframe in-place
    df_out = df.copy()

    # 1. Prevent extremely high severity values by clipping the minimum SPEI reading to -5.0
    df_out['spei_adj'] = df_out[spei_col].clip(lower=-5.0)

    # 2. Create the 'drought' column (1 if below threshold, 0 otherwise)
    df_out['drought'] = (df_out['spei_adj'] < threshold).astype(int)

    # Structure for the output dictionary
    results_dict = {}
    target_rps = [1, 3, 5, 10, 20, 30, 50, 100]

    def fit_distribution(data, total_years):
        """Helper to fit Gamma distribution and compute exact quantities for return periods"""
        num_events = len(data)
        if num_events < 2:
            # Need at least two events to confidently fit a distribution variance, otherwise return NaN
            return {str(rp): np.nan for rp in target_rps}

        # Lambda is the events per year frequency
        lambda_rate = num_events / total_years

        # Fit a Gamma distribution (forces loc=0 to ensure we only get positive sizes back)
        try:
            shape, loc, scale = gamma.fit(data, floc=0)
        except Exception:
            return {str(rp): np.nan for rp in target_rps}

        res = {}
        for rp in target_rps:
            # If the calculated return period is shorter than the average time it normally takes
            # for ANY event to appear, it mathematically shouldn't resolve (e.g. asking for a 1-year event
            # when droughts naturally only happen every 2 years). We set it to 0.0.
            if lambda_rate * rp <= 1.0:
                res[str(rp)] = 0.0
            else:
                # Calculate the Peak Over Threshold (POT) equivalent probability
                # p = 1 - (1 / (lambda * Return Period))
                p = 1.0 - (1.0 / (lambda_rate * rp))
                val = gamma.ppf(p, shape, loc=loc, scale=scale)
                res[str(rp)] = round(float(val), 2)
        return res

    # 3. Group the dataset by each unique combination of geography and climate model
    grouped = df_out.groupby(group_cols)

    for (ecoregion, scenario, gcm), group in grouped:
        # Guarantee it's chronologically forward
        group = group.sort_values(by=['year', 'month'])

        # We find total years safely by taking the count of months divided by 12 (since we know it is continuous)
        total_years = len(group) / 12.0

        # 4. Group adjacent months recursively into unified Drought Events
        # This will flag True whenever the drought switches from 0 to 1, or 1 to 0
        is_drought = group['drought'] == 1
        event_id_flags = (is_drought != is_drought.shift(1)).cumsum()

        drought_periods = group[is_drought].copy()

        if len(drought_periods) == 0:
            # If the particular scenario literally never got below the threshold, log blanks
            dur_rps = {str(rp): 0.0 for rp in target_rps}
            int_rps = {str(rp): 0.0 for rp in target_rps}
        else:
            drought_periods['event_id'] = event_id_flags[is_drought]

            # 5. Extract duration & cumulative severity per event block
            event_stats = drought_periods.groupby('event_id').agg(
                duration=('drought', 'count'),  # Continuous amount of months
                intensity=('spei_adj', lambda monthly_vals: abs(monthly_vals.sum()))  # Cumulative magnitude
            ).reset_index()

            # Predict our statistical distributions
            dur_rps = fit_distribution(event_stats['duration'].values, total_years)
            int_rps = fit_distribution(event_stats['intensity'].values, total_years)

        # 6. Populate the nested dictionary object cleanly
        if ecoregion not in results_dict:
            results_dict[ecoregion] = {}
        if scenario not in results_dict[ecoregion]:
            results_dict[ecoregion][scenario] = {}

        results_dict[ecoregion][scenario][gcm] = {
            "duration": dur_rps,
            "intensity": int_rps
        }

    # Drop the temporary clipping calculation so DataFrame matches the original dimensions
    df_out.drop(columns=['spei_adj'], inplace=True)

    return df_out, results_dict