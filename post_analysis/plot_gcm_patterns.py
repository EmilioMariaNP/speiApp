import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger

# Add project root to path for utils import if needed
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
    
    # Resolve copula analysis CSV path
    copula_csv = config.get("copula_analysis_csv", "copula_analysis/nuts_2_copula_analysis.csv")
    if home_dir and not os.path.isabs(copula_csv):
        copula_csv = os.path.join(home_dir, copula_csv)
        
    # Resolve post analysis plots directory from config
    post_plots_dir = config.get("post_analysis_plots_dir", "post_analysis/plots")
    if not os.path.isabs(post_plots_dir):
        post_plots_dir = os.path.join(PROJECT_ROOT, post_plots_dir)
        
    os.makedirs(post_plots_dir, exist_ok=True)
    return config, copula_csv, post_plots_dir

def setup_plotting_theme():
    sns.set_theme(style="whitegrid", font_scale=1.1)
    plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
    plt.rcParams['axes.edgecolor'] = '#cccccc'
    plt.rcParams['axes.linewidth'] = 1.0

def load_and_prep_data(data_path):
    df = pd.read_csv(data_path)
    slice_cols = ['NUTS', 'scenario', 'spei', 'Return period']
    
    # Slice ensemble baselines
    df['ensemble_median'] = df.groupby(slice_cols)['Likelihood_Change'].transform('median')
    eps = 1e-6
    df['ratio_to_median'] = df['Likelihood_Change'] / np.maximum(df['ensemble_median'], eps)
    df['gcm_rank'] = df.groupby(slice_cols)['Likelihood_Change'].rank(ascending=False, method='min')
    
    return df

def plot_01_global_bias_distributions(df, plots_dir):
    plt.figure(figsize=(12, 7))
    
    # Custom palette
    gcm_colors = {
        'UKESM1-0-LL': '#d73027',
        'IPSL-CM6A-LR': '#fc8d59',
        'MRI-ESM2-0': '#fee090',
        'MPI-ESM1-2-HR': '#91bfdb',
        'GFDL-ESM4': '#4575b4'
    }
    
    # Sort GCMs by median ratio descending
    gcm_order = df.groupby('gcm')['ratio_to_median'].median().sort_values(ascending=False).index.tolist()
    palette = [gcm_colors.get(g, '#808080') for g in gcm_order]
    
    ax = sns.boxplot(
        data=df,
        x='gcm',
        y='ratio_to_median',
        hue='gcm',
        order=gcm_order,
        palette=palette,
        fliersize=3,
        linewidth=1.2,
        width=0.5,
        legend=False
    )
    
    ax.set_yscale('log')
    ax.axhline(1.0, color='black', linestyle='--', linewidth=1.5, label='Ensemble Median (1.0x)')
    
    ax.set_title("Global Distribution of GCM Prediction Ratio (Relative to Ensemble Median)", fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel("Global Circulation Model (GCM)", fontsize=12, fontweight='bold')
    ax.set_ylabel("Likelihood Change / Ensemble Median (Log Scale)", fontsize=12, fontweight='bold')
    
    # Add median text annotations
    for i, gcm in enumerate(gcm_order):
        med_val = df[df['gcm'] == gcm]['ratio_to_median'].median()
        ax.text(i, med_val * 1.15, f"Median:\n{med_val:.2f}x", ha='center', va='bottom', fontsize=10, fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", lw=0.5, alpha=0.85))
        
    ax.legend(loc='upper right', frameon=True)
    plt.tight_layout()
    
    out_file = os.path.join(plots_dir, "01_gcm_global_bias_distributions.png")
    plt.savefig(out_file, dpi=300)
    plt.close()
    logger.info(f"Saved plot: {out_file}")

def plot_02_country_heatmap_matrix(df, plots_dir):
    plt.figure(figsize=(14, 10))
    
    # Pivot table of country vs gcm median ratio
    ctry_pivot = df.groupby(['Country', 'gcm'])['ratio_to_median'].median().unstack()
    
    # Order countries by UKESM1-0-LL ratio descending (or spatial geography)
    if 'UKESM1-0-LL' in ctry_pivot.columns:
        ctry_order = ctry_pivot['UKESM1-0-LL'].sort_values(ascending=False).index
        ctry_pivot = ctry_pivot.loc[ctry_order]
        
    # Reorder GCM columns by global bias
    gcm_order = ['UKESM1-0-LL', 'IPSL-CM6A-LR', 'MRI-ESM2-0', 'MPI-ESM1-2-HR', 'GFDL-ESM4']
    existing_gcms = [g for g in gcm_order if g in ctry_pivot.columns]
    ctry_pivot = ctry_pivot[existing_gcms]
    
    ax = sns.heatmap(
        ctry_pivot,
        annot=True,
        fmt=".2f",
        cmap="YlOrRd",
        cbar_kws={'label': 'Median Ratio to Ensemble Median'},
        linewidths=0.8,
        linecolor='white'
    )
    
    ax.set_title("Geographical Heatmap: GCM Relative Prediction Ratio by Country", fontsize=15, fontweight='bold', pad=15)
    ax.set_xlabel("Global Circulation Model (GCM)", fontsize=13, fontweight='bold')
    ax.set_ylabel("Country (Spatial Unit Group)", fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    out_file = os.path.join(plots_dir, "02_gcm_country_heatmap_matrix.png")
    plt.savefig(out_file, dpi=300)
    plt.close()
    logger.info(f"Saved plot: {out_file}")

def plot_03_return_period_divergence(df, plots_dir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6.5))
    
    gcm_colors = {
        'UKESM1-0-LL': '#d73027',
        'IPSL-CM6A-LR': '#fc8d59',
        'MRI-ESM2-0': '#e6ab02',
        'MPI-ESM1-2-HR': '#4575b4',
        'GFDL-ESM4': '#313695'
    }
    
    # 1. Absolute Likelihood Change by Return Period
    rp_abs = df.groupby(['Return period', 'gcm'])['Likelihood_Change'].median().reset_index()
    sns.lineplot(
        data=rp_abs,
        x='Return period',
        y='Likelihood_Change',
        hue='gcm',
        palette=gcm_colors,
        marker='o',
        linewidth=2.5,
        markersize=8,
        ax=ax1
    )
    ax1.set_title("(A) Absolute Median Likelihood Change by Return Period", fontsize=13, fontweight='bold')
    ax1.set_xlabel("Return Period (Years)", fontsize=12, fontweight='bold')
    ax1.set_ylabel("Median Likelihood Change", fontsize=12, fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    # 2. Ratio to Ensemble Median by Return Period
    rp_rel = df.groupby(['Return period', 'gcm'])['ratio_to_median'].median().reset_index()
    sns.lineplot(
        data=rp_rel,
        x='Return period',
        y='ratio_to_median',
        hue='gcm',
        palette=gcm_colors,
        marker='s',
        linewidth=2.5,
        markersize=8,
        ax=ax2
    )
    ax2.axhline(1.0, color='black', linestyle='--', linewidth=1.5, label='Ensemble Median')
    ax2.set_title("(B) Relative Prediction Ratio vs Return Period", fontsize=13, fontweight='bold')
    ax2.set_xlabel("Return Period (Years)", fontsize=12, fontweight='bold')
    ax2.set_ylabel("Median Ratio to Ensemble Median", fontsize=12, fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.6)
    
    plt.suptitle("Amplification of GCM Prediction Spread Across Return Periods", fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    out_file = os.path.join(plots_dir, "03_gcm_return_period_divergence.png")
    plt.savefig(out_file, dpi=300)
    plt.close()
    logger.info(f"Saved plot: {out_file}")

def plot_04_spei_timescale_comparison(df, plots_dir):
    plt.figure(figsize=(11, 6.5))
    
    spei_gcm = df.groupby(['gcm', 'spei'])['ratio_to_median'].median().reset_index()
    spei_gcm['spei_label'] = spei_gcm['spei'].apply(lambda x: f"SPEI-{x}")
    
    gcm_order = ['UKESM1-0-LL', 'IPSL-CM6A-LR', 'MRI-ESM2-0', 'MPI-ESM1-2-HR', 'GFDL-ESM4']
    
    ax = sns.barplot(
        data=spei_gcm,
        x='gcm',
        y='ratio_to_median',
        hue='spei_label',
        order=gcm_order,
        palette=['#e74c3c', '#3498db'],
        edgecolor='black',
        linewidth=0.8
    )
    
    ax.axhline(1.0, color='black', linestyle='--', linewidth=1.5, label='Ensemble Median (1.0x)')
    ax.set_title("GCM Prediction Ratios: Short-Term (SPEI-3) vs Long-Term (SPEI-12) Droughts", fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel("Global Circulation Model (GCM)", fontsize=12, fontweight='bold')
    ax.set_ylabel("Median Ratio to Ensemble Median", fontsize=12, fontweight='bold')
    
    # Annotate bars
    for p in ax.patches:
        height = p.get_height()
        if not np.isnan(height) and height > 0:
            ax.annotate(f"{height:.2f}x",
                        (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom',
                        fontsize=10, fontweight='bold',
                        xytext=(0, 3), textcoords='offset points')
            
    ax.legend(title="Drought Timescale", frameon=True)
    plt.tight_layout()
    
    out_file = os.path.join(plots_dir, "04_gcm_spei_timescale_comparison.png")
    plt.savefig(out_file, dpi=300)
    plt.close()
    logger.info(f"Saved plot: {out_file}")

def plot_05_rank_frequency_stacked(df, plots_dir):
    plt.figure(figsize=(12, 6.5))
    
    # Calculate percentage of each rank (1 to 5) per GCM
    rank_counts = df.groupby(['gcm', 'gcm_rank']).size().unstack(fill_value=0)
    rank_pcts = rank_counts.div(rank_counts.sum(axis=1), axis=0) * 100
    
    gcm_order = ['UKESM1-0-LL', 'IPSL-CM6A-LR', 'MRI-ESM2-0', 'MPI-ESM1-2-HR', 'GFDL-ESM4']
    rank_pcts = rank_pcts.loc[[g for g in gcm_order if g in rank_pcts.index]]
    
    rank_colors = ['#d73027', '#f46d43', '#fee08b', '#d9ef8b', '#4575b4']
    
    ax = rank_pcts.plot(
        kind='bar',
        stacked=True,
        figsize=(12, 6.5),
        color=rank_colors,
        edgecolor='black',
        linewidth=0.8
    )
    
    ax.set_title("Frequency Breakdown of GCM Prediction Ranks across Spatial Units", fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel("Global Circulation Model (GCM)", fontsize=12, fontweight='bold')
    ax.set_ylabel("Percentage of Total Slices (%)", fontsize=12, fontweight='bold')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    
    # Legend labels
    ax.legend(
        title="Prediction Rank",
        labels=['Rank 1 (Highest)', 'Rank 2', 'Rank 3', 'Rank 4', 'Rank 5 (Lowest)'],
        bbox_to_anchor=(1.02, 1),
        loc='upper left',
        frameon=True
    )
    
    # Add text percentages inside stacked bars if >= 8%
    for c in ax.containers:
        for p in c:
            height = p.get_height()
            if height >= 8.0:
                ax.annotate(f"{height:.1f}%",
                            (p.get_x() + p.get_width() / 2., p.get_y() + height / 2.),
                            ha='center', va='center',
                            fontsize=9.5, fontweight='bold', color='black')
                
    plt.tight_layout()
    out_file = os.path.join(plots_dir, "05_gcm_rank_frequency_stacked.png")
    plt.savefig(out_file, dpi=300)
    plt.close()
    logger.info(f"Saved plot: {out_file}")

def main(config_path=None):
    setup_plotting_theme()
    config, copula_csv, post_plots_dir = resolve_paths(config_path)
    
    logger.info(f"Loading data from: {copula_csv}")
    logger.info(f"Saving plots to: {post_plots_dir}")
    
    df = load_and_prep_data(copula_csv)
    plot_01_global_bias_distributions(df, post_plots_dir)
    plot_02_country_heatmap_matrix(df, post_plots_dir)
    plot_03_return_period_divergence(df, post_plots_dir)
    plot_04_spei_timescale_comparison(df, post_plots_dir)
    plot_05_rank_frequency_stacked(df, post_plots_dir)
    
    logger.info("All 5 post-analysis plots successfully generated!")

if __name__ == "__main__":
    cfg_arg = sys.argv[1] if len(sys.argv) > 1 else None
    main(cfg_arg)
