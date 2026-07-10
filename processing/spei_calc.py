import os
import sys
import pandas as pd
import numpy as np
import scipy.stats as sps
from loguru import logger

# Ensure project root is in python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.utils import load_config, save_dataframe, generate_ai_commentary
from utils.spei_lib import calculate_spei_indices

def spei_calc_main(config):
    logger.info("Starting Stage 1: SPEI Calculation")

    spatial_unit_col = config.get("spatial_unit_col", "spatial_unit")
    target_sus = config.get("spatial_units", [])
    
    val_csv = config.get("validation_dataset")
    proj_csv = config.get("projections_csv")
    
    if not val_csv or not os.path.exists(val_csv):
        logger.error(f"Validation dataset not found at {val_csv}")
        sys.exit(-1)
        
    if not proj_csv or not os.path.exists(proj_csv):
        logger.error(f"Projections dataset not found at {proj_csv}")
        sys.exit(-1)
        
    logger.info("Loading datasets...")
    df_val = pd.read_csv(val_csv)
    df_proj = pd.read_csv(proj_csv)

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
            # For observed, we fit and transform on the entire series
            # calculate_spei_indices expects 'date' column
            df_val_su['date'] = pd.to_datetime(df_val_su['date'])
            df_val_su_spei_only = calculate_spei_indices(
                df=df_val_su,
                spei_list=spei_indices,
                wb_col='water_balance',
                date_col='date',
                dist=sps.fisk
            )
            # Merge back the original columns
            df_val_su_spei = pd.merge(
                df_val_su, 
                df_val_su_spei_only.drop(columns=['water_balance'], errors='ignore'), 
                on='date', how='left'
            )
            df_val_su_spei['gcm'] = np.nan
            dfs_spei_results.append(df_val_su_spei)
        
        # 2. Modeled GCM Data
        df_proj_su = df_proj[df_proj[spatial_unit_col] == su].copy()
        if not df_proj_su.empty:
            gcms = df_proj_su['gcm'].dropna().unique()
            df_proj_su['date'] = pd.to_datetime(df_proj_su['date'])
            
            for gcm in gcms:
                logger.info(f"  Fitting Modeled Data for {su} - GCM: {gcm}...")
                df_gcm = df_proj_su[df_proj_su['gcm'] == gcm].copy()
                
                df_gcm_ref = df_gcm[df_gcm['scenario'] == ref_scenario]
                if df_gcm_ref.empty:
                    logger.warning(f"  No reference scenario ({ref_scenario}) found for {gcm} in {su}. Fitting on all available data.")
                    start_yr = None
                    end_yr = None
                else:
                    start_yr = int(df_gcm_ref['year'].min())
                    end_yr = int(df_gcm_ref['year'].max())
                
                df_gcm_spei_only = calculate_spei_indices(
                    df=df_gcm,
                    spei_list=spei_indices,
                    wb_col='water_balance',
                    date_col='date',
                    start_year=start_yr,
                    end_year=end_yr,
                    dist=sps.fisk
                )
                
                df_gcm_spei = pd.merge(
                    df_gcm, 
                    df_gcm_spei_only.drop(columns=['water_balance'], errors='ignore'), 
                    on='date', how='left'
                )
                dfs_spei_results.append(df_gcm_spei)
            
    # Combine everything
    if dfs_spei_results:
        logger.info("Combining results and saving...")
        df_spei_final = pd.concat(dfs_spei_results, ignore_index=True)
        
        out_csv = config.get("spei_csv", "Outputs/spei_data.csv")
        os.makedirs(os.path.dirname(out_csv) or '.', exist_ok=True)
        
        save_dataframe(df_spei_final, out_csv)
        
        # Generate AI commentary
        out_dir = os.path.dirname(out_csv) or '.'
        comment_text = (
            f"Successfully calculated SPEI indices {spei_indices} for {len(target_sus)} spatial units. "
            f"Fitted Reference scenario ({ref_scenario}) and transformed to all projections. "
            f"Observed data fitted independently."
        )
        generate_ai_commentary(summary_text=comment_text, output_dir=out_dir, stage_name="Stage_1")
        
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
