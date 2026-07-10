import os
import sys
import pandas as pd
import numpy as np
import scipy.stats as st
from loguru import logger

# Ensure project root is in python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.utils import load_config, save_dataframe, generate_ai_commentary



def calculate_spei(wb_series, scale=1):
    """
    Calculates SPEI by fitting the distribution to the entirety of the provided data.

    Parameters:
    wb_series (pd.Series): Monthly water balance (PR - PET) with a DatetimeIndex.
    scale (int): The accumulation timescale (e.g., 1 for 1-month, 3 for 3-month).

    Returns:
    pd.Series: Calculated SPEI values.
    """
    wb_series.index = pd.to_datetime(wb_series.index)

    # 1. Handle the accumulation scale
    if scale > 1:
        data = wb_series.rolling(scale).sum().dropna()
    else:
        data = wb_series.dropna()

    # Empty series to store the results
    spei_result = pd.Series(index=data.index, dtype=float)

    # 2. Loop through each of the 12 months to handle seasonality
    for month in range(1, 13):

        # Extract data strictly for this calendar month
        month_data = data[data.index.month == month]

        # Skip if there isn't enough data to fit a distribution
        if len(month_data) < 2:
            continue

        # 3. Fit the Fisk (Log-Logistic) distribution
        params = st.fisk.fit(month_data)

        # 4. Calculate the Cumulative Distribution Function (CDF)
        cdf = st.fisk.cdf(month_data, *params)

        # 5. Transform the CDF to standard normal deviate (SPEI)
        # Clipping prevents 0 or 1 probabilities from becoming -inf or inf
        cdf = np.clip(cdf, 1e-6, 1 - 1e-6)
        spei_result.loc[month_data.index] = st.norm.ppf(cdf)

    return spei_result.sort_index()


def calculate_continuous_spei(wb_hist, wb_proj, scale=1):
    """
    wb_hist: Monthly historical water balance (pandas Series with DatetimeIndex)
    wb_proj: Monthly future water balance (pandas Series with DatetimeIndex)
    scale: The SPEI time scale (1 uses raw monthly data; 3 accumulates 3 months, etc.)
    """
    wb_hist.index = pd.to_datetime(wb_hist.index)
    wb_proj.index = pd.to_datetime(wb_proj.index)

    # 1. Handle the Accumulation Scale
    if scale > 1:
        hist_data = wb_hist.rolling(scale).sum().dropna()
        proj_data = wb_proj.rolling(scale).sum().dropna()
    else:
        hist_data = wb_hist.dropna()
        proj_data = wb_proj.dropna()

    # Empty series to store the results
    spei_hist = pd.Series(index=hist_data.index, dtype=float)
    spei_proj = pd.Series(index=proj_data.index, dtype=float)

    # 2. Loop through each of the 12 months to account for seasonality
    for month in range(1, 13):

        hist_month = hist_data[hist_data.index.month == month]
        proj_month = proj_data[proj_data.index.month == month]

        if len(hist_month) < 2:
            continue

        # 3. Fit the Fisk (Log-Logistic) distribution on the HISTORICAL data
        params = st.fisk.fit(hist_month)

        # 4. Apply parameters to the HISTORICAL data itself
        cdf_hist = st.fisk.cdf(hist_month, *params)
        cdf_hist = np.clip(cdf_hist, 1e-6, 1 - 1e-6)
        spei_hist.loc[hist_month.index] = st.norm.ppf(cdf_hist)

        # 5. Apply those same parameters to the FUTURE data
        if len(proj_month) > 0:
            cdf_proj = st.fisk.cdf(proj_month, *params)
            cdf_proj = np.clip(cdf_proj, 1e-6, 1 - 1e-6)
            spei_proj.loc[proj_month.index] = st.norm.ppf(cdf_proj)

    # 6. Combine into a single DataFrame
    # df_hist = pd.DataFrame({'spei': spei_hist, 'period': 'historical'})
    # df_proj = pd.DataFrame({'spei': spei_proj, 'period': 'projected'})
    #df_combined = pd.concat([df_hist, df_proj]).sort_index()

    return spei_hist, spei_proj



def spei_calc_main(config):
    logger.info("Starting Stage 1: SPEI Calculation")

    spatial_unit_col = config.get("spatial_unit_col", "spatial_unit")
    target_sus = config.get("spatial_units", [])
    
    val_csv = config.get("validation_dataset")
    proj_csv = config.get("projections_csv")

    start_yr = config['ref_start_year']
    end_yr = config['ref_end_year']

    proj_start_yr = config["proj_start_year"]
    proj_end_yr = config["proj_end_year"]

    
    if not val_csv or not os.path.exists(val_csv):
        logger.error(f"Validation dataset not found at {val_csv}")
        sys.exit(-1)
        
    if not proj_csv or not os.path.exists(proj_csv):
        logger.error(f"Projections dataset not found at {proj_csv}")
        sys.exit(-1)
        
    logger.info("Loading datasets...")
    df_val = pd.read_csv(val_csv, parse_dates=['date'])
    df_proj = pd.read_csv(proj_csv, parse_dates=['date'])

    # Ensure validation data has the missing columns as empty/null
    for col in ['scenario', 'gcm']: 
        if col not in df_val.columns:
            df_val[col] = np.nan
            
    # Filter by spatial_units if provided
    if target_sus and len(target_sus) > 0:
        logger.info(f"Filtering for spatial units: {target_sus}")
        df_val = df_val[df_val[spatial_unit_col].isin(target_sus)]
        df_proj = df_proj[df_proj[spatial_unit_col].isin(target_sus)]
    else:
        logger.info("No spatial_units provided, processing all available units.")
        target_sus = pd.concat([df_val[spatial_unit_col], df_proj[spatial_unit_col]]).dropna().unique().tolist()

    spei_indices = config.get("spei_indices", [3, 12])
    # The JSON config may provide indices as ints (e.g. 3, 12) or strings ('spei3')
    spei_indices = [int(str(x).replace('spei', '')) for x in spei_indices]

    ref_scenario = config.get("reference_scenario", "historical")
    
    dfs_spei_results = []

    for su in target_sus:
        logger.info(f"Calculating SPEI indices for spatial unit: {su}")
        
        # 1. Observed Data
        df_val_su = df_val[df_val[spatial_unit_col] == su].copy()
        if not df_val_su.empty:
            logger.info(f"  Fitting Observed Data for {su}...")

            if start_yr is not None and end_yr is not None:
                logger.info(f'Clipping validation dataset to years {start_yr}-{end_yr}')
                df_val_su = df_val_su[(df_val_su['year'] >= start_yr) & (df_val_su['year'] <= end_yr)]


            df_val_su.set_index('date', inplace = True)

            for spei_scale in spei_indices:
                wb_validation = df_val_su['water_balance'].copy(deep=True)
                spei_validation = calculate_spei(wb_validation, scale=spei_scale)
                df_val_su[f'spei{spei_scale}'] = spei_validation

            df_val_su['scenario'] = 'validation'
            dfs_spei_results.append(df_val_su.reset_index())


        # 2. Modeled GCM Data
        df_proj_su = df_proj[df_proj[spatial_unit_col] == su].copy()
        if not df_proj_su.empty:
            gcms = df_proj_su['gcm'].dropna().unique().tolist()
            scenarios = df_proj_su['scenario'].dropna().unique().tolist()
            scenarios.remove(ref_scenario)

            for gcm in gcms:
                df_hist = df_proj_su[(df_proj_su['scenario'] == ref_scenario) & (df_proj_su['gcm'] == gcm)].copy(deep=True)
                if start_yr is not None and end_yr is not None:
                    logger.info(f'Clipping {su}-{gcm} historical dataset to years {start_yr}-{end_yr}')
                    df_hist = df_hist[(df_hist['year'] >= start_yr) & (df_hist['year'] <= end_yr)]

                df_hist.set_index('date', inplace=True)

                for scenario in scenarios:
                    logger.info(f'Fitting SPEI for {su}-{scenario}-{gcm}')

                    df_proj_scenario = df_proj_su[(df_proj_su['scenario'] == scenario) & (df_proj_su['gcm'] == gcm)].copy(deep=True)

                    if proj_start_yr is not None and proj_end_yr is not None:
                        logger.info(f'Clipping projections dataset to years {proj_start_yr}-{proj_end_yr}')
                        df_proj_scenario = df_proj_scenario[(df_proj_scenario['year'] >= proj_start_yr) & (df_proj_scenario['year'] <= proj_end_yr)]

                    df_proj_scenario.set_index('date', inplace=True)

                    for spei_scale in spei_indices:

                        wb_hist = df_hist['water_balance'].copy(deep=True)
                        wb_proj = df_proj_scenario['water_balance'].copy(deep=True)

                        spei_hist, spei_proj = calculate_continuous_spei(wb_hist, wb_proj, scale=spei_scale)

                        df_proj_scenario[f'spei{spei_scale}'] = spei_proj

                        if f'spei{spei_scale}' not in df_hist.columns: # avoid adding historical-gcm spei multiple times
                            df_hist[f'spei{spei_scale}'] = spei_hist

                    dfs_spei_results.append(df_proj_scenario.reset_index())

                dfs_spei_results.append(df_hist.reset_index()) # append historical-gcm dataframe only once, before exiting the gcm iteration
                logger.info(f'SPEI {spei_indices} calculated for {su}-{gcm}')

    # Combine everything
    if dfs_spei_results:
        logger.info("Combining results and saving...")
        df_spei_final = pd.concat(dfs_spei_results, ignore_index=True)

        df_spei_final['date'] = pd.to_datetime(df_spei_final['date']) # make date format uniform

        for spei_scale in spei_indices:
            df_spei_final[f'spei{spei_scale}'] = round(df_spei_final[f'spei{spei_scale}'], 2)

        out_csv = config.get("spei_csv", "Outputs/spei_data.csv")
        os.makedirs(os.path.dirname(out_csv) or '.', exist_ok=True)
        
        save_dataframe(df_spei_final, out_csv)

        logger.info("Stage 1 execution completed.")
    else:
        logger.warning("No data processed!")

if __name__ == "__main__":
    # Standard entrypoint loading the generic config json
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'config.json')
    config_dict = load_config(config_path)
    if config_dict:
        spei_calc_main(config_dict)
    else:
        logger.error("Failed to load config.json. Exiting.")
