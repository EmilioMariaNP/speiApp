from processing.plots import plot_dists
from utils.spei_lib import calculate_spei_indices
from utils.utils import load_config, filter_by_overlapping_years, is_same_distribution, save_dataframe, merge_dataframes
import pandas as pd
import numpy as np
import scipy.stats as stats
import sys
import os
from loguru import logger
import json
import scipy.stats as sps
import os

# Ensure project root is in python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.utils import generate_ai_commentary


def compute_rolling_balance(df, scale):
    """Compute rolling water balance for a specific timescale (min_periods=scale for strict matching)."""
    # Sort dataframe by Year and Month before calculating rolling sum
    df_sorted = df.sort_values(["Year", "Month"]).copy()
    rolling_sum = (
        df_sorted["water_balance"].rolling(window=scale, min_periods=scale).sum()
    )

    # Restore original index using sort_index (requires index was unique)
    # The rolling preserves index.
    return rolling_sum.reindex(df.index)


def fit_and_transform_spei(fit_data, transform_data):
    """
    Fits a probability distribution to fit_data and transforms transform_data into SPEI.
    Uses Fisk (Log-logistic) as it is standard for SPEI.
    """
    # Fit data: remove nan
    valid_fit = fit_data.dropna()
    if len(valid_fit) < 10:  # Minimum data threshold for basic stats fitting
        logger.warning(
            "Insufficient data to fit SPEI distribution (less than 10 non-null values)."
        )
        return pd.Series(np.nan, index=transform_data.index)

    try:
        # Fit Log-Logistic (Fisk) distribution
        c, loc, scale = stats.fisk.fit(valid_fit)

        # Transform data (get CDF)
        cdf = stats.fisk.cdf(transform_data.dropna(), c, loc=loc, scale=scale)

        # Avoid infinity at extreme probability bounds
        cdf = np.clip(cdf, 0.0001, 0.9999)

        # Convert cumulative probabilities to standard normal (SPEI values)
        spei_vals = stats.norm.ppf(cdf)

        # Re-assign mapping back to the full original series index
        result_series = pd.Series(np.nan, index=transform_data.index)
        result_series.loc[transform_data.dropna().index] = spei_vals
        return result_series

    except Exception as e:
        logger.error(f"Error fitting SPEI distribution: {str(e)}")
        return pd.Series(np.nan, index=transform_data.index)


def spei_calc_main(config):
    logger.info("Starting Stage 1: SPEI Calculation")

    target_sus = config.get("spatial_units", [])

    ks_records = [] # holds the results of the ks tests
    dfs_spei = []

    # Load Validation (Observed) Dataset
    if validation_csv and os.path.exists(validation_csv):
        logger.info(f"Loading Validation dataset: {validation_csv}")
        df_val_sus = pd.read_csv(validation_csv)
    else:
        logger.error(f"Validation dataset not found at {validation_csv}, program will exit.")
        exit(-1)

    # Load Projections (Modeled) Dataset
    if proj_csv and os.path.exists(proj_csv):
        logger.info(f"Loading Projections dataset: {proj_csv}")
        df_proj = pd.read_csv(proj_csv)
        df_proj['dataset'] = df_proj['gcm']
    else:
        logger.error(f"Projections dataset not found at {proj_csv}, program will exit.")
        exit(-1)

    scenarios = df_proj['scenario'].unique().tolist()
    gcms = df_proj['gcm'].unique().tolist()


    if len(target_sus) == 0:
        target_sus = df_proj[spatial_unit_col].unique().tolist()

    for target_su in target_sus:

        logger.info(f'Comparing projection data vs {target_su} validation datasets..')
        # use only the reference (historical) scenario for the current region/spatial unit
        df_ref = df_proj[(df_proj['scenario'] == ref_scenario) &
                         (df_proj[spatial_unit_col] == target_su)
        ]

        df_val = df_val_sus.copy(deep=True)
        df_val = df_val[df_val['region'] == target_su]

        # make sure the datasets times overlap
        df_val, df_ref = filter_by_overlapping_years(df_a=df_val,
                                                     df_b=df_ref
                                                     )

        logger.info(f'Comparing {len(df_ref.index)} vs {len(df_val.index)} rows.')

        df_val['dataset'] = val_dataset_label
        df_val = df_val[['date', 'year', 'month', 'dataset', 'water_balance']]
        df_ref_concat = df_ref[['date', 'year', 'month', 'dataset', 'water_balance']].copy(deep=True)

        df = pd.concat([df_val, df_ref_concat], ignore_index=True)


        # if show_plots:
        #     plot_dists(df=df,
        #                fig_title=f'{val_dataset_label.upper()} vs GCMs - {selected_region}',
        #                x_col='water_balance',
        #                hue_col='dataset',
        #                )

        # calculate spei
        dfs_gcms = []
        for gcm in gcms:
            # calculate spei indices for each gcm in the reference period
            logger.info(f'Calculating SPEI indices for {gcm}...')
            df_ref_gcm = df_ref[df_ref['gcm'] == gcm]
            df_ref_spei = calculate_spei_indices(df=df_ref_gcm,
                                                 spei_list= spei_scales,
                                                 wb_col='water_balance',
                                                 date_col='date',
                                                 start_year=None,
                                                 end_year=None,
                                                 dist=sps.fisk
                                                 )
            df_ref_spei['dataset'] = gcm
            dfs_gcms.append(df_ref_spei)

        df_gcms = pd.concat(dfs_gcms, ignore_index=True)

        # calculate spei indices for the validation dataset
        df_val_spei = calculate_spei_indices(df=df_val,
                                             spei_list=spei_scales,
                                             wb_col='water_balance',
                                             date_col='date',
                                             start_year=None,
                                             end_year=None,
                                             dist=sps.fisk
                                             )
        df_val_spei['dataset'] = val_dataset_label

        df_spei = pd.concat([df_val_spei, df_gcms], ignore_index=True)
        df_spei['month'] = df_spei['date'].dt.month


        for spei_scale in spei_scales:
            spei_n_col = f'spei{spei_scale}'
            df_test = df_spei.copy(deep=True)
            if spei_scale == 3: # plot and use only summer spei3
                df_test = df_spei[df_spei['month'].isin([3, 4, 5, 6, 7, 8, 9])]  # plot and use only summer spei3
            # Plot spei
            if save_plots:
                fig_file = f'spe{spei_scale}_{target_su}_validation.png'
                fig_file = os.path.join(validation_plots_dir, fig_file)
                plot_dists(df=df_test,
                           fig_title=f'SPEI-{spei_scale} {val_dataset_label.upper()} vs GCMs - {target_su}',
                           x_col= spei_n_col,
                           hue_col='dataset',
                           x_label=f'SPEI-{spei_scale}',
                           show_fig = show_plots,
                           out_file=fig_file
                           )

            # test if gcms are consistent with the observed data
            spei_n_val = df_test[df_test['dataset'] == val_dataset_label][spei_n_col]
            for gcm in gcms:
                spei_n_gcm = df_test[df_test['dataset'] == gcm][spei_n_col]
                ks_res, p_val = is_same_distribution(ts1 = spei_n_val ,
                                                     ts2 = spei_n_gcm
                                                     )
                record = {
                    'test dataset': val_dataset_label,
                    'region': target_su,
                    'spei': spei_scale,
                    'gcm': gcm,
                    'KS passed': ks_res,
                    'p-value': round(p_val, 3)
                }
                ks_records.append(record)
                logger.info(f'KS test for {target_su}-{gcm}- passed: {ks_res} (p-value: {round(p_val, 3)})')

    df_ks_test = pd.DataFrame(ks_records)
    save_dataframe(df=df_ks_test, csv_file=ks_csv)


    # calculate spei for gcm datasets (incl. reference period)
    dfs_spei = []
    for target_su in target_sus:
        logger.info(f'Calculating SPEI indices for {target_su}...')
        dfs_su = []
        for spei_scale in spei_scales:
            dfs_scale = []

            # reliable gcms for this spei scale-spatial unit
            reliable_gcms = df_ks_test[(df_ks_test[spatial_unit_col] == target_su) &
                                       (df_ks_test['KS passed'] == True) &
                                       (df_ks_test['spei'] == spei_scale)]['gcm'].tolist()
            for gcm in gcms:
                for scenario in scenarios:
                    df_reg_gcm = df_proj[(df_proj[spatial_unit_col] == target_su) &
                                        (df_proj['gcm'] == gcm) &
                                        (df_proj['scenario'].isin([ref_scenario, scenario]))].copy(deep=True)

                    df_spei_n_reg = calculate_spei_indices(df=df_reg_gcm,
                                                           spei_list=[spei_scale],
                                                           wb_col='water_balance',
                                                           date_col='date',
                                                           start_year=ref_start_year,
                                                           end_year=ref_end_year,
                                                           dist=sps.fisk
                                                           )
                    df_spei_n_reg['gcm'] = gcm
                    df_spei_n_reg['scenario'] = scenario
                    df_spei_n_reg[spatial_unit_col] = target_su
                    df_spei_n_reg[f'spei{spei_scale}_valid'] = False
                    if gcm in reliable_gcms: # update the spei dataframe to mark the "reliable" gcms
                        df_spei_n_reg[f'spei{spei_scale}_valid'] = True

                    dfs_scale.append(df_spei_n_reg)

            df_scale = pd.concat(dfs_scale, ignore_index=True)
            dfs_su.append(df_scale)

        # merge all spei of this spatial unit
        df_su = merge_dataframes(df_list = dfs_su,
                                 group_cols = ['date', spatial_unit_col, 'scenario', 'gcm', 'water_balance' ],
                                 how='inner')

        dfs_spei.append(df_su)

    df_spei = pd.concat(dfs_spei, ignore_index=True)
    df_spei['month'] = df_spei['date'].dt.month
    df_spei['year'] = df_spei['date'].dt.year
    save_dataframe(df=df_spei,csv_file=spei_csv)







    #
    # # Filtering for spatial units
    # if target_sus and len(target_sus) > 0:
    #     df = df[df[su_col].isin(target_sus)]
    #     logger.info(f"Filtered dataset for specified spatial units: {target_sus}")
    #
    # spei_indices = config.get("spei_indices", ["spei3"])
    # ref_scenario = config.get("reference_scenario", "reference")
    #
    # # Initialize index columns with NaN
    # for idx_name in spei_indices:
    #     df[idx_name] = np.nan
    #
    # spatial_units_list = df[su_col].unique()
    #
    # for su in spatial_units_list:
    #     logger.info(f"Processing Spatial Unit: {su}")
    #     su_mask = df[su_col] == su
    #
    #     for idx_name in spei_indices:
    #         # Extract timescale scale from names like "spei3"
    #         try:
    #             scale = int(idx_name.replace("spei", ""))
    #         except ValueError:
    #             logger.error(f"Invalid SPEI index format '{idx_name}'. Skipping.")
    #             continue
    #
    #         rolling_col = f"wb_rolling_{scale}"
    #
    #         # Compute rolling water balance for the specific scale
    #         df.loc[su_mask, rolling_col] = compute_rolling_balance(
    #             df.loc[su_mask], scale
    #         )
    #
    #         # Sub-masks: Observed vs Modeled
    #         # Based on Index Fitting Rule 1 and 2
    #         obs_mask = su_mask & (df["gcm"].isna() | (df["gcm"] == ""))
    #         gcm_mask = su_mask & ~(df["gcm"].isna() | (df["gcm"] == ""))
    #
    #         # 1. Observed Data: Fit and Transform on itself
    #         if obs_mask.any():
    #             fit_data = df.loc[obs_mask, rolling_col]
    #             transform_data = df.loc[obs_mask, rolling_col]
    #             df.loc[obs_mask, idx_name] = fit_and_transform_spei(
    #                 fit_data, transform_data
    #             )
    #
    #         # 2. Modeled GCM Data: Fit on reference scenario, transform all scenarios per GCM
    #         if gcm_mask.any():
    #             gcms = df.loc[gcm_mask, "gcm"].unique()
    #             for gcm in gcms:
    #                 gcm_specific_mask = gcm_mask & (df["gcm"] == gcm)
    #                 ref_mask = gcm_specific_mask & (df["scenario"] == ref_scenario)
    #
    #                 if ref_mask.sum() == 0:
    #                     logger.warning(
    #                         f"No reference scenario '{ref_scenario}' found for GCM '{gcm}' in {su}. Cannot fit distribution."
    #                     )
    #                     continue
    #
    #                 fit_data = df.loc[ref_mask, rolling_col]
    #                 transform_data = df.loc[gcm_specific_mask, rolling_col]
    #
    #                 df.loc[gcm_specific_mask, idx_name] = fit_and_transform_spei(
    #                     fit_data, transform_data
    #                 )
    #
    #         # Cleanup dummy rolling column
    #         df = df.drop(columns=[rolling_col], errors="ignore")
    #
    # output_path = config.get("spei_csv", "Outputs/spei_data.csv")
    # os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # df.to_csv(output_path, index=False)
    # logger.info(f"Stage 1 complete. Computed SPEI indices saved to {output_path}")
    #
    # # Generate results commentary representing insights
    # generate_ai_commentary(
    #     summary_text=f"Calculated SPEI distributions for indices {spei_indices} across {len(spatial_units_list)} spatial units.",
    #     output_dir=os.path.dirname(output_path),
    #     stage_name="spei_calc",
    # )


if __name__ == "__main__":
    # Standard entrypoint loading the generic config json
    config_path = '../data/config.json'
    config = load_config(config_path)


    ref_scenario = config['reference_scenario']
    selected_region = config['spatial_units']
    spatial_unit_col = config['spatial_unit_col']
    spei_scales = config['spei_indices']

    ref_start_year = config["ref_start_year"]
    ref_end_year = config["ref_end_year"]

    save_plots = True
    show_plots = False

    proj_csv = config['projections_csv']
    validation_csv = config['validation_dataset']

    ks_csv = config['ks_csv']
    validation_plots_dir = config['validation_plots_dir']
    spei_csv = config['spei_csv']

    val_dataset_label = 'validation'

    spei_calc_main(config)










