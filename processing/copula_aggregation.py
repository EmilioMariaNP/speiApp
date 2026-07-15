"""
Aggregates by weighted average the expected likelihood changes.
"""

import os
import pandas as pd
from loguru import logger
import numpy as np

from utils.utils import load_config, load_dataframe, save_dataframe


def main_copula_aggregation(config):

    logger.info("Starting copula aggregation...")

    spei_scales = config.get("spei_indices", [3, 12])
    copula_csv = config.get("copula_analysis_csv", None)
    return_periods_csv = config["return_periods_csv"]
    spatial_unit_col = config["spatial_unit_col"]

    df = load_dataframe(copula_csv)

    return_periods = df["Return period"].unique().tolist()
    spatial_units = df[spatial_unit_col].unique().tolist()
    scenarios = df["scenario"].unique().tolist()

    # calculate reliability as sum of fit scores
    df["reliability"] = (
        df["Ref_duration_fit_score"]
        + df["Ref_intensity_fit_score"]
        + df["Proj_duration_fit_score"]
        + df["Proj_intensity_fit_score"]
    )

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
        else:
            w_sum = np.sum(w_valid)
            if w_sum > 0:
                weighted_mean = np.sum(vals_valid * w_valid) / w_sum
            else:
                weighted_mean = np.mean(vals_valid)

            std_val = np.std(vals_valid, ddof=1) if len(vals_valid) > 1 else 0.0
            median_val = np.median(vals_valid)

            if np.isnan(weighted_mean) or weighted_mean == 0:
                cv_val = np.nan
            else:
                cv_val = std_val / weighted_mean

        return pd.Series(
            {
                "Likelihood_Change": weighted_mean,
                "Likelihood_Change_std": std_val,
                "Likelihood_Change_median": median_val,
                "Likelihood_Change_cv": cv_val,
            }
        )

    try:
        df_rp = (
            df.groupby(["Return period", "scenario", spatial_unit_col])
            .apply(calculate_weighted_stats, include_groups=False)
            .reset_index()
        )
    except TypeError:
        df_rp = (
            df.groupby(["Return period", "scenario", spatial_unit_col])
            .apply(calculate_weighted_stats)
            .reset_index()
        )

    df_rp.sort_values(by=[spatial_unit_col, "scenario", "Return period"], inplace=True)
    df_rp["Likelihood_Change"] = round(df_rp["Likelihood_Change"], 1)
    df_rp["Likelihood_Change_std"] = round(df_rp["Likelihood_Change_std"], 2)
    df_rp["Likelihood_Change_median"] = round(df_rp["Likelihood_Change_median"], 1)
    df_rp["Likelihood_Change_cv"] = round(df_rp["Likelihood_Change_cv"], 2)

    save_dataframe(df_rp, return_periods_csv)


if __name__ == "__main__":
    config_file = "../data/config_nut2.json"
    # config_file = "../data/config_ecoregions.json"

    config = load_config(config_file)

    home_dir = config.get("home_dir", None)
    if home_dir:

        config["return_periods_csv"] = os.path.join(
            home_dir, config["return_periods_csv"]
        )
        config["copula_analysis_csv"] = os.path.join(
            home_dir, config["copula_analysis_csv"]
        )

    main_copula_aggregation(config)

    print("Process completed.")
