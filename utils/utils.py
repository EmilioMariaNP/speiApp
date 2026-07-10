import json
import os
from loguru import logger
import numpy as np
import pandas as pd
from scipy import stats
import scipy.stats as sps
from functools import reduce


def load_config(config_path="config.json"):
    """Load the JSON configuration file."""
    if not os.path.exists(config_path):
        logger.error(f"Configuration file {config_path} not found.")
        return {}
    with open(config_path, 'r') as f:
        return json.load(f)

def filter_by_overlapping_years(df_a, df_b):
    # get overlapping years and filter by
    y_max = np.amin([df_a['year'].max(), df_b['year'].max()])
    y_min = np.amax([df_a['year'].min(), df_b['year'].min()])
    # print(y_max, y_min)

    df_a = df_a[(df_a['year'] >= y_min) & (df_a['year'] <= y_max)]
    df_b = df_b[(df_b['year'] >= y_min) & (df_b['year'] <= y_max)]

    return df_a, df_b

def save_dataframe(df, csv_file):
    df.to_csv(csv_file, index=False)
    logger.info(f"Successfully saved dataframe to {csv_file}")
    return 0

def load_dataframe(csv_file):
    df = pd.read_csv(csv_file)
    logger.info(f"{csv_file} loaded")
    return df

def merge_dataframes(df_list, group_cols, how='inner'):
    if group_cols is None:
        group_cols = ['date', 'region', 'scenario', 'gcm', 'water_balance']
    df = None
    if len(df_list) > 0:
        if len(df_list) == 1:
            df = df_list[0]
        else:
            df = reduce(
                lambda left, right: pd.merge(left, right, on=group_cols,
                                             how=how), df_list)

    return df



def is_same_distribution(ts1: pd.Series, ts2: pd.Series, alpha: float = 0.05) -> bool:
    """
    Applies a Two-Sample Kolmogorov-Smirnov (K-S) test to assess if two
    pandas time series are drawn from the same distribution.

    Args:
        ts1 (pd.Series): First time series.
        ts2 (pd.Series): Second time series.
        alpha (float): Significance level threshold (default is 0.05).

    Returns:
        bool: True if they appear to be from the same distribution
              (fails to reject the null hypothesis), False otherwise.
    """

    # 1. Drop NaN values since the K-S test cannot process missing data
    data1 = ts1.dropna()
    data2 = ts2.dropna()

    # Safety check
    if data1.empty or data2.empty:
        raise ValueError("One or both time series are empty after dropping NaNs.")

    # 2. Perform the Two-Sample K-S test
    stat, p_value = stats.ks_2samp(data1, data2)

    # 3. Evaluate the p-value against the alpha threshold
    # The null hypothesis is that the two series are drawn from the same distribution.
    # If p_value > alpha, we fail to reject the null hypothesis (return True).
    same_dist = bool(p_value > alpha)
    return same_dist, p_value


def generate_ai_commentary(summary_text, output_dir, stage_name):
    """
    Generate an AI-written commentary (Stub implementation).
    """
    logger.info(f"Generating AI commentary for {stage_name}...")
    output_filename = os.path.join(output_dir, f"{stage_name}_results_ai_comments.txt")
    
    os.makedirs(output_dir, exist_ok=True)
    
    commentary = (
        f"AI Commentary for {stage_name}:\n"
        f"{summary_text}\n"
        f"Note: This is an auto-generated placeholder commentary. In a full implementation, "
        f"this would integrate with an LLM API."
    )
    
    with open(output_filename, 'w') as f:
        f.write(commentary)
    
    logger.info(f"Commentary saved to {output_filename}")



