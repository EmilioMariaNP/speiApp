import pandas as pd
import numpy as np
import os
import sys
from loguru import logger

# Ensure project root is in python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.utils import load_config

def extract_drought_events(df, threshold, spei_col):
    """
    Extracts drought events from a pandas DataFrame containing 'year' 'month' and SPEI values.
    Returns a dataframe of events with duration and severity.
    """
    if df.empty:
        return pd.DataFrame(columns=['duration', 'severity'])
        
    is_dry = df[spei_col] <= threshold
    
    # Calculate absolute integer representing the month globally to verify continuity
    abs_month = df['year'] * 12 + df['month']
    
    # Check if the previous row in the sorted dataframe is exactly 1 month prior
    is_contiguous = (abs_month - abs_month.shift(1)) == 1
    
    # A new event starts when it's dry AND (it wasn't previously dry OR there is a gap in months)
    start_of_new_event = is_dry & (~is_dry.shift(1).fillna(False) | ~is_contiguous)
    
    # Create an ID for each sequence of dry months
    event_ids = start_of_new_event.cumsum()
    
    # Extract only the true dry periods using boolean masking
    dry_periods = df[is_dry].copy()
    
    if dry_periods.empty:
        return pd.DataFrame(columns=['duration', 'severity'])
        
    # Group by the dynamically created consecutive block IDs
    dry_event_ids = event_ids[is_dry]
    
    events = dry_periods.groupby(dry_event_ids).agg(
        duration=(spei_col, 'count'),
        severity=(spei_col, lambda x: np.abs(np.sum(x)))
    )
    
    return events

def events_count_main(config):
    logger.info("Starting Stage 3: Drought Events Extraction")
    
    spei_csv = config.get("spei_csv")
    if not spei_csv or not os.path.exists(spei_csv):
        logger.error(f"Input file not found: {spei_csv}")
        sys.exit(1)
        
    logger.info(f"Loading SPEI calculated datasets from: {spei_csv}")
    df = pd.read_csv(spei_csv)
    
    # Load Basic Configuration Params
    su_col = config.get("spatial_unit_col", "region")
    threshold = config.get("spei_threshold", -0.5)
    spei_indices = config.get("spei_indices", [3, 12])
    spei_3_months = config.get("spei_3_months", [5, 6, 7, 8])
    
    # Load Expected Output File Templates
    dry_events_csv_tmpl = config.get("dry_events_csv")
    dry_events_len_csv_tmpl = config.get("dry_events_len_csv")
    dry_events_sev_csv_tmpl = config.get("dry_events_sev_csv")
    
    for scale in spei_indices:
        spei_col = f"spei{scale}"
        valid_col = f"spei{scale}_valid"
        
        logger.info(f"Processing Events for Index: {spei_col}")
        if spei_col not in df.columns:
            logger.warning(f"{spei_col} column not found in input data. Skipping.")
            continue
            
        # 1. Restrict events entirely to Valid GCM/Scenarios using Stage 2 ks-test validations
        if valid_col in df.columns:
            # We filter for rows evaluated as specifically True. 
            scale_df = df[df[valid_col] == True].copy()
            logger.info(f"Filtered {len(df) - len(scale_df)} rows based on valid GCM constraint for {spei_col}.")
        else:
            logger.warning(f"Validation marker column {valid_col} not found! Extracting broadly.")
            scale_df = df.copy()
            
        scale_df = scale_df.dropna(subset=[spei_col])
        if scale_df.empty:
            logger.warning(f"No valid data remaining for {spei_col} after dataset filters.")
            continue
            
        # 2. For Agricultural scales (spei3), limit months explicitly
        if scale == 3 and 'month' in scale_df.columns:
            logger.info(f"Limiting agricultural scale 3 to specified months: {spei_3_months}")
            scale_df = scale_df[scale_df['month'].isin(spei_3_months)]
            
        counts_records = []
        durations_records = []
        severities_records = []
        
        # Assume missing labels implicitly refer to 'Observed'
        scale_df['scenario'] = scale_df['scenario'].fillna('Observed')
        scale_df['gcm'] = scale_df['gcm'].fillna('Observed')
        
        if 'year' in scale_df.columns and 'month' in scale_df.columns:
            scale_df = scale_df.sort_values(by=[su_col, 'scenario', 'gcm', 'year', 'month'])
        else:
            logger.error("Required 'year' and 'month' columns not fully present for temporal continuity.")
            continue
            
        grouped = scale_df.groupby([su_col, 'scenario', 'gcm'])
        
        for (su, scenario, gcm), group in grouped:
            
            # 1. Total Count (Total dry months matching the threshold for this partition)
            dry_count = (group[spei_col] <= threshold).sum()
            counts_records.append({
                su_col: su,
                'scenario': scenario,
                'gcm': gcm,
                'dry_months_count': int(dry_count),
                'total_months': len(group)
            })
            
            # 2 & 3. Extract Distinct Drought Event characteristics (Duration, Severity)
            events_df = extract_drought_events(group, threshold, spei_col)
            
            if not events_df.empty:
                for _, row in events_df.iterrows():
                    durations_records.append({
                        su_col: su,
                        'scenario': scenario,
                        'gcm': gcm,
                        'duration': int(row['duration'])
                    })
                    severities_records.append({
                        su_col: su,
                        'scenario': scenario,
                        'gcm': gcm,
                        'severity': float(row['severity'])
                    })
                    
        # Define Output Paths dynamically mapping the scale list
        count_csv = dry_events_csv_tmpl.replace("{index}", str(scale))
        len_csv = dry_events_len_csv_tmpl.replace("{index}", str(scale))
        sev_csv = dry_events_sev_csv_tmpl.replace("{index}", str(scale))
        
        # Ensure structural integrity corresponding to filepaths
        os.makedirs(os.path.dirname(count_csv), exist_ok=True)
        os.makedirs(os.path.dirname(len_csv), exist_ok=True)
        os.makedirs(os.path.dirname(sev_csv), exist_ok=True)
        
        # Export processed frames cleanly
        pd.DataFrame(counts_records).to_csv(count_csv, index=False)
        logger.info(f"Saved total counts CSV format to -> {count_csv}")
        
        if durations_records:
            pd.DataFrame(durations_records).to_csv(len_csv, index=False)
        else:
            pd.DataFrame(columns=[su_col, 'scenario', 'gcm', 'duration']).to_csv(len_csv, index=False)
        logger.info(f"Saved duration vectors CSV sequence to -> {len_csv}")
            
        if severities_records:
            pd.DataFrame(severities_records).to_csv(sev_csv, index=False)
        else:
            pd.DataFrame(columns=[su_col, 'scenario', 'gcm', 'severity']).to_csv(sev_csv, index=False)
        logger.info(f"Saved severity metrics CSV distribution to -> {sev_csv}")

if __name__ == "__main__":

    # Handle the JSON path cleanly whether executing locally from the /processing dir or globally
    config_path = '../data/config.json'
    if not os.path.exists(config_path):
        config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data/config.json")
    
    if not os.path.exists(config_path):
        logger.error(f"Cannot discover JSON setting path securely using fallback root at {config_path}")
        sys.exit(1)
        
    config = load_config(config_path)
    
    events_count_main(config)
    logger.info("Stage 3 completed effectively.")
