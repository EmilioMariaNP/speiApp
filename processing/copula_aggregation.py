"""
Aggregates by weighted average the expected likelihood changes.
"""

import os
import pandas as pd
from loguru import logger
import numpy as np

from utils.utils import load_config, load_dataframe, save_dataframe


# def main_copula_aggregation(config):
#
#     logger.info("Starting copula aggregation...")
#
#     copula_csv = config.get("copula_analysis_csv", None)
#     copula_analysis_aggregation_csv = config["copula_analysis_aggregation_csv"]
#     spatial_unit_col = config["spatial_unit_col"]
#
#     df = load_dataframe(copula_csv)
#
#     # average out the likelihood change from the different GCMS weighted by reliability
#
#     def calculate_weighted_stats(group):
#         vals = group["Likelihood_Change"].values
#         w = group["reliability"].values
#
#         valid = ~(np.isnan(vals) | np.isnan(w))
#         vals_valid = vals[valid]
#         w_valid = w[valid]
#
#         if len(vals_valid) == 0:
#             weighted_mean = np.nan
#             std_val = np.nan
#             median_val = np.nan
#             cv_val = np.nan
#         else:
#             w_sum = np.sum(w_valid)
#             if w_sum > 0:
#                 weighted_mean = np.sum(vals_valid * w_valid) / w_sum
#             else:
#                 weighted_mean = np.mean(vals_valid)
#
#             std_val = np.std(vals_valid, ddof=1) if len(vals_valid) > 1 else 0.0
#             median_val = np.median(vals_valid)
#
#             if np.isnan(weighted_mean) or weighted_mean == 0:
#                 cv_val = np.nan
#             else:
#                 cv_val = std_val / weighted_mean
#
#         return pd.Series(
#             {
#                 "Likelihood_Change": weighted_mean,
#                 "Likelihood_Change_std": std_val,
#                 "Likelihood_Change_median": median_val,
#                 "Likelihood_Change_cv": cv_val,
#             }
#         )
#
#     try:
#         df_rp = (
#             df.groupby(["Return period", "scenario", 'spei', spatial_unit_col])
#             .apply(calculate_weighted_stats, include_groups=False)
#             .reset_index()
#         )
#     except TypeError:
#         df_rp = (
#             df.groupby(["Return period", "scenario", spatial_unit_col, 'spei'])
#             .apply(calculate_weighted_stats)
#             .reset_index()
#         )
#
#     df_rp.sort_values(by=[spatial_unit_col, "scenario", 'spei', "Return period"], inplace=True)
#     df_rp["Likelihood_Change"] = round(df_rp["Likelihood_Change"], 1)
#     df_rp["Likelihood_Change_std"] = round(df_rp["Likelihood_Change_std"], 2)
#     df_rp["Likelihood_Change_median"] = round(df_rp["Likelihood_Change_median"], 1)
#     df_rp["Likelihood_Change_cv"] = round(df_rp["Likelihood_Change_cv"], 2)
#
#     save_dataframe(df_rp, copula_analysis_aggregation_csv)


def main_copula_aggregation(config):

    logger.info("Starting copula aggregation...")

    copula_csv = config.get("copula_analysis_csv", None)
    copula_analysis_aggregation_csv = config["copula_analysis_aggregation_csv"]
    spatial_unit_col = config["spatial_unit_col"]

    df = load_dataframe(copula_csv)

    if config.get('strict_aggregation', False): # is strict_aggregation, only fittings acceptable goodness of fit for ref and proj are considered

        # filter out the rows where at least one fit is not reliable
        logger.info('Filtering unrealiable fits...')
        n_rows = len(df.index)
        df = df[(df['Ref_duration_fit_score'] != 0) &
                (df['Ref_intensity_fit_score'] != 0) &
                (df['Proj_duration_fit_score'] != 0) &
                (df['Proj_intensity_fit_score'] != 0)
        ].copy(deep=True)
        df['reliability'] = 1 # set to 1 to avoid changing the weighting function TODO: update

        n_rows_filtered = len(df.index)
        delta_rows = n_rows - n_rows_filtered
        logger.info(f'Original dataframe size: {n_rows}, filtered rows: {n_rows_filtered} ({delta_rows} droppped)')

    # average out the likelihood change from the different GCMS weighted by reliability

    def calculate_weighted_stats(group):
        vals = group["Likelihood_Change"].values
        w = group["reliability"].values

        valid = ~(np.isnan(vals) | np.isnan(w))
        vals_valid = vals[valid]
        w_valid = w[valid]

        if len(vals_valid) == 0:
            weighted_mean = np.nan
            std_val = np.nan
            median_val = np.nan
            cv_val = np.nan
            iqr_val = np.nan
            mad_val = np.nan
            rel_mad_val = np.nan
        else:
            w_sum = np.sum(w_valid)
            if w_sum > 0:
                weighted_mean = np.sum(vals_valid * w_valid) / w_sum
            else:
                weighted_mean = np.mean(vals_valid)

            std_val = np.std(vals_valid, ddof=1) if len(vals_valid) > 1 else 0.0
            median_val = np.median(vals_valid)
            iqr_val = np.percentile(vals_valid, 75) - np.percentile(vals_valid, 25)
            mad_val = np.median(np.abs(vals_valid - median_val))


            if np.isnan(weighted_mean) or weighted_mean == 0:
                cv_val = np.nan
            else:
                cv_val = round(std_val / weighted_mean, 2)

            if np.isnan(median_val) or median_val == 0:
                rel_mad_val = np.nan
            else:
                rel_mad_val = round(mad_val / median_val, 2)

        return pd.Series(
            {
                "Likelihood_Change_mean": weighted_mean,
                "Likelihood_Change_std": std_val,
                "Likelihood_Change_median": median_val,
                "Likelihood_Change_cv": cv_val,
                "Likelihood_Change_iqr": iqr_val,
                "Likelihood_Change_mad": mad_val,
                "Likelihood_Change_rel_mad": rel_mad_val
            }
        )

    try:
        df_rp = (
            df.groupby(["Return period", "scenario", 'spei', spatial_unit_col])
            .apply(calculate_weighted_stats, include_groups=False)
            .reset_index()
        )
    except TypeError:
        df_rp = (
            df.groupby(["Return period", "scenario", spatial_unit_col, 'spei'])
            .apply(calculate_weighted_stats)
            .reset_index()
        )

    df_rp.sort_values(by=[spatial_unit_col, "scenario", 'spei', "Return period"], inplace=True)
    if "Likelihood_Change" in df_rp.columns:
        df_rp["Likelihood_Change"] = round(df_rp["Likelihood_Change"], 1)
    if "Likelihood_Change_mean" in df_rp.columns:
        df_rp["Likelihood_Change_mean"] = round(df_rp["Likelihood_Change_mean"], 1)
    df_rp["Likelihood_Change_std"] = round(df_rp["Likelihood_Change_std"], 2)
    df_rp["Likelihood_Change_median"] = round(df_rp["Likelihood_Change_median"], 1)
    df_rp["Likelihood_Change_cv"] = round(df_rp["Likelihood_Change_cv"], 2)
    df_rp["Likelihood_Change_iqr"] = round(df_rp["Likelihood_Change_iqr"], 2)
    df_rp["Likelihood_Change_mad"] = round(df_rp["Likelihood_Change_mad"], 2)

    save_dataframe(df_rp, copula_analysis_aggregation_csv)





if __name__ == "__main__":
    config_file = "../data/config_nut2.json"
    # config_file = "../data/config_ecoregions.json"

    config = load_config(config_file)

    home_dir = config.get("home_dir", None)
    if home_dir:

        config["return_periods_csv"] = os.path.join(
            home_dir, config["return_periods_csv"]
        )
        config["copula_analysis_aggregation_csv"] = os.path.join(
            home_dir, config["copula_analysis_aggregation_csv"]
        )

    main_copula_aggregation(config)

    print("Process completed.")
