
import pandas as pd
import os
from loguru import logger
from scipy import stats

from utils.utils import load_config, load_dataframe, dump_json_to_file


def compare_distributions_ks(series1, series2, alpha=0.05):
    """
    Performs a two-sample Kolmogorov-Smirnov test on two pandas Series.

    Parameters:
    series1 (pd.Series): First data series
    series2 (pd.Series): Second data series
    alpha (float): Significance level (default is 0.05)

    Returns:
    tuple: (ks_statistic, p_value, is_same_distribution)
    """
    # 1. Clean the data by dropping any missing values
    s1 = series1.dropna()
    s2 = series2.dropna()

    # 2. Perform the two-sample KS test
    ks_stat, p_value = stats.ks_2samp(s1, s2)

    # 3. Interpret the result
    # Null Hypothesis (H0): The two samples are drawn from the same distribution.
    # If p_value > alpha, we fail to reject H0 (they are likely the same)
    is_same_distribution = bool(p_value > alpha)

    return ks_stat, p_value, is_same_distribution


def main_gcms_eval(config):
    """
    Compare by means of KS test the historical-gcm SPEI-n with the remote sensing derived SPEI-n by spatial unit to test if the gcms replicate
    the observed period.
    """

    spatial_unit_col = config.get("spatial_unit_col", "spatial_unit")
    target_sus = config.get("spatial_units", [])
    spei_indices = config.get("spei_indices", [3, 12])
    spei_cols = [f'spei{x}' for x in spei_indices]

    reference_scenario = config.get("reference_scenario", 'historical')

    df_spei = load_dataframe(config.get("spei_csv", "Outputs/spei_data.csv"))
    df_spei['date'] = pd.to_datetime(df_spei['date'])

    df_val = df_spei[df_spei['scenario'] == 'validation']
    df_hist = df_spei[df_spei['scenario'] == reference_scenario]

    gcms = df_hist['gcm'].unique().tolist()

    if len(target_sus)==0: # get all spatial units if no target is specified
        target_sus = df_spei[spatial_unit_col].unique().tolist()

    results = []

    spei_threshold = float(config.get('spei_threshold', -1))

    for su in target_sus:
        df_val_su = df_val[df_val[spatial_unit_col] == su].sort_values('date')
        for spei_col in spei_cols:
            val_spei_n = round(df_val_su[df_val_su[spei_col]<=spei_threshold][spei_col], 1)
            for gcm in gcms:
                logger.info(f'Processing {su}-{gcm}-{spei_col}...')
                df_hist_su_gcm = df_hist[(df_hist[spatial_unit_col] == su) &
                                         (df_hist['gcm'] == gcm)
                ].sort_values('date')

                gcm_spei_n = round(df_hist_su_gcm[df_hist_su_gcm[spei_col]<=spei_threshold][spei_col], 1)

                stat, p, is_same = compare_distributions_ks(gcm_spei_n, val_spei_n)
                logger.info(f"{su}-{gcm}-{spei_col} -> Stat: {stat:.4f}, p-value: {p:.4f}, Same Dist: {is_same}")

                record = {
                    spatial_unit_col: su,
                    'gcm': gcm,
                    'spei': spei_col,
                    'ks-test': round(stat, 3),
                    'p-value': round(p, 4),
                    'passed': is_same
                }

                results.append(record)

    config['valid_gcms'] = results

    return config



if __name__ == "__main__":
    # Standard entrypoint loading the generic config json
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'config_nut2.json')
    config_dict = load_config(config_path)
    if config_dict:
        config_dict = main_gcms_eval(config_dict)
        dump_json_to_file(config_dict, config_path, indent=4)

    else:
        logger.error("Failed to load config_nut2.json. Exiting.")