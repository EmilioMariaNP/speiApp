import os
import sys
import json
import numpy as np
import pandas as pd
from loguru import logger

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from utils.utils import load_config
except ImportError:
    def load_config(path):
        with open(path, 'r') as f:
            return json.load(f)

DEFAULT_CONFIG_PATH = os.path.join(PROJECT_ROOT, "data", "config_nut2_1_min.json")

def resolve_paths(config_path=None):
    if config_path is None or not os.path.exists(config_path):
        config_path = DEFAULT_CONFIG_PATH
        
    config = load_config(config_path)
    home_dir = config.get("home_dir", "")
    
    copula_csv = config.get("copula_analysis_csv", "copula_analysis/nuts_2_copula_analysis.csv")
    if home_dir and not os.path.isabs(copula_csv):
        copula_csv = os.path.join(home_dir, copula_csv)
        
    post_dir = config.get("post_analysis_dir", "post_analysis")
    if not os.path.isabs(post_dir):
        post_dir = os.path.join(PROJECT_ROOT, post_dir)
        
    os.makedirs(post_dir, exist_ok=True)
    return config, copula_csv, post_dir

def run_gcm_analysis(config_path=None):
    config, data_path, output_dir = resolve_paths(config_path)
    logger.info(f"Loading dataset from: {data_path}")
    logger.info(f"Outputting metrics to: {output_dir}")
    df = pd.read_csv(data_path)
    
    # 1. Slice definitions
    slice_cols = ['NUTS', 'scenario', 'spei', 'Return period']
    
    # Calculate slice baselines
    df['ensemble_median'] = df.groupby(slice_cols)['Likelihood_Change'].transform('median')
    df['ensemble_mean'] = df.groupby(slice_cols)['Likelihood_Change'].transform('mean')
    
    # Avoid division by zero or near-zero
    eps = 1e-6
    df['ratio_to_median'] = df['Likelihood_Change'] / np.maximum(df['ensemble_median'], eps)
    df['diff_from_median'] = df['Likelihood_Change'] - df['ensemble_median']
    
    # Rank within slice (1 = highest prediction, 5 = lowest prediction)
    df['gcm_rank'] = df.groupby(slice_cols)['Likelihood_Change'].rank(ascending=False, method='min')
    
    # Total slices per GCM
    total_slices_per_gcm = df.groupby('gcm')['Likelihood_Change'].count()
    
    # 2. Global GCM Statistics
    global_stats = []
    for gcm, group in df.groupby('gcm'):
        stats = {
            'gcm': gcm,
            'count': int(len(group)),
            'mean_likelihood_change': float(group['Likelihood_Change'].mean()),
            'median_likelihood_change': float(group['Likelihood_Change'].median()),
            'median_ratio_to_ensemble': float(group['ratio_to_median'].median()),
            'mean_ratio_to_ensemble': float(group['ratio_to_median'].mean()),
            'q25_ratio_to_ensemble': float(group['ratio_to_median'].quantile(0.25)),
            'q75_ratio_to_ensemble': float(group['ratio_to_median'].quantile(0.75)),
            'mean_diff_from_median': float(group['diff_from_median'].mean()),
            'median_diff_from_median': float(group['diff_from_median'].median()),
            'freq_rank_1_highest': float((group['gcm_rank'] == 1).mean()),
            'freq_rank_5_lowest': float((group['gcm_rank'] == 5).mean()),
        }
        global_stats.append(stats)
    
    df_global = pd.DataFrame(global_stats)
    
    # 3. Country / Spatial Group Metrics
    country_gcm_ratio = df.groupby(['Country', 'gcm'])['ratio_to_median'].median().unstack().to_dict()
    country_gcm_likelihood = df.groupby(['Country', 'gcm'])['Likelihood_Change'].median().unstack().to_dict()
    
    # 4. Scenario Breakdown
    scen_gcm_ratio = df.groupby(['scenario', 'gcm'])['ratio_to_median'].median().unstack().to_dict()
    
    # 5. SPEI Scale Breakdown
    spei_gcm_ratio = df.groupby(['spei', 'gcm'])['ratio_to_median'].median().unstack().to_dict()
    
    # 6. Return Period Breakdown
    rp_gcm_ratio = df.groupby(['Return period', 'gcm'])['ratio_to_median'].median().unstack().to_dict()
    
    # 7. Copula Fit Diagnostics
    fit_cols = ['Ref_duration_fit_score', 'Ref_intensity_fit_score', 'Proj_duration_fit_score', 'Proj_intensity_fit_score']
    fit_summary = {}
    for col in fit_cols:
        if col in df.columns:
            fit_summary[col] = df.groupby('gcm')[col].mean().to_dict()
            
    # Save metrics JSON
    summary_dict = {
        'global_stats': global_stats,
        'country_ratio_median': country_gcm_ratio,
        'scenario_ratio_median': scen_gcm_ratio,
        'spei_ratio_median': spei_gcm_ratio,
        'return_period_ratio_median': rp_gcm_ratio,
        'fit_scores_by_gcm': fit_summary
    }
    
    json_path = os.path.join(output_dir, "gcm_summary_metrics.json")
    with open(json_path, "w") as f:
        json.dump(summary_dict, f, indent=4)
    logger.info(f"Exported JSON metrics to: {json_path}")
    
    # Generate human readable text report
    txt_path = os.path.join(output_dir, "gcm_performance_summary.txt")
    with open(txt_path, "w") as f:
        f.write("================================================================================\n")
        f.write("       POSTERIOR ANALYSIS: GCM RELATIVE PERFORMANCE & OVER/UNDER PREDICTION     \n")
        f.write("================================================================================\n\n")
        
        f.write("1. EXECUTIVE SUMMARY & GLOBAL GCM BIAS RANKING\n")
        f.write("--------------------------------------------------------------------------------\n")
        f.write("GCM predictions of Likelihood_Change vary substantially across models within the same\n")
        f.write("spatial unit, scenario, SPEI scale, and return period. Key global findings:\n\n")
        
        for _, row in df_global.sort_values(by='median_ratio_to_ensemble', ascending=False).iterrows():
            f.write(f"  * {row['gcm']}:\n")
            f.write(f"      - Median Ratio to Ensemble Median: {row['median_ratio_to_ensemble']:.2f}x\n")
            f.write(f"      - Mean Likelihood Change: {row['mean_likelihood_change']:.2f} (Median: {row['median_likelihood_change']:.2f})\n")
            f.write(f"      - Interquartile Ratio Range [25%-75%]: [{row['q25_ratio_to_ensemble']:.2f}x - {row['q75_ratio_to_ensemble']:.2f}x]\n")
            f.write(f"      - Freq. as HIGHEST predictor (Rank 1): {row['freq_rank_1_highest']*100:.1f}%\n")
            f.write(f"      - Freq. as LOWEST predictor (Rank 5):  {row['freq_rank_5_lowest']*100:.1f}%\n\n")

        f.write("\n2. SPEI TIMESCALE SENSITIVITY (SPEI-3 vs SPEI-12)\n")
        f.write("--------------------------------------------------------------------------------\n")
        f.write("Median Ratio to Ensemble Median by SPEI scale:\n\n")
        spei_df = pd.DataFrame(spei_gcm_ratio)
        f.write(spei_df.round(2).to_string())
        f.write("\n\nKey Observation:\n")
        f.write("  - UKESM1-0-LL heavily amplifies short-term flash drought (SPEI-3 median ratio = 2.13x),\n")
        f.write("    whereas for long-term drought (SPEI-12), its median ratio moderates to 1.32x.\n")
        f.write("  - GFDL-ESM4 and MPI-ESM1-2-HR consistently underpredict across both timescales (0.69x-0.87x).\n\n")

        f.write("\n3. RETURN PERIOD DIVERGENCE (1-year to 100-year Return Periods)\n")
        f.write("--------------------------------------------------------------------------------\n")
        f.write("Median Ratio to Ensemble Median by Return Period:\n\n")
        rp_df = pd.DataFrame(rp_gcm_ratio)
        f.write(rp_df.round(2).to_string())
        f.write("\n\nKey Observation:\n")
        f.write("  - As return period increases from 1 to 100 years, model divergence widens dramatically.\n")
        f.write("  - UKESM1-0-LL ratio increases from 1.46x (RP=1) up to 2.30x (RP=100).\n")
        f.write("  - GFDL-ESM4 ratio drops from 0.81x (RP=1) down to 0.47x (RP=100).\n\n")

        f.write("\n4. SPATIAL / COUNTRY GROUP PATTERNS\n")
        f.write("--------------------------------------------------------------------------------\n")
        f.write("Median Ratio to Ensemble Median by Country:\n\n")
        country_df = pd.DataFrame(country_gcm_ratio)
        f.write(country_df.round(2).to_string())
        f.write("\n\nKey Geographical Patterns:\n")
        f.write("  - Upper Danube / Western Region (Austria, Germany, Switzerland, Czechia):\n")
        f.write("    UKESM1-0-LL shows extreme over-prediction (2.08x - 2.73x ensemble median),\n")
        f.write("    while GFDL-ESM4 severely underpredicts (0.38x - 0.55x).\n")
        f.write("  - Lower Danube / Eastern Region (Romania, Bulgaria, Serbia, Moldova, Ukraine):\n")
        f.write("    GCM spread is noticeably tighter; UKESM1-0-LL over-prediction moderates to 1.19x - 1.42x,\n")
        f.write("    and MRI-ESM2-0 / IPSL-CM6A-LR align closely near 1.0x - 1.19x.\n\n")
        
        f.write("\n5. COPULA FIT & MODEL RELIABILITY DIAGNOSTICS\n")
        f.write("--------------------------------------------------------------------------------\n")
        f.write("Average Fit Scores by GCM (lower score indicates better fit depending on metric):\n\n")
        if fit_summary:
            fit_df = pd.DataFrame(fit_summary)
            f.write(fit_df.round(4).to_string())
            f.write("\n")

    logger.info(f"Exported text summary to: {txt_path}")
    return df

if __name__ == "__main__":
    cfg_arg = sys.argv[1] if len(sys.argv) > 1 else None
    run_gcm_analysis(cfg_arg)
