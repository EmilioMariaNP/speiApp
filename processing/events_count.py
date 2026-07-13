import pandas as pd
import numpy as np
import os
import sys
from loguru import logger

# Ensure project root is in python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.utils import load_config


def compute_drought_events(df, dry_col = 'spei3_dry', spei_col = 'spei3'):
    """
    Computes the length and intensity of drought events for each GCM
    using a single-month pooling strategy.
    """
    # Ensure the dataframe is chronologically sorted per GCM
    df = df.sort_values(by=['gcm', 'year', 'month']).reset_index(drop=True)

    all_gcm_events = []

    # Process each GCM group independently
    for gcm_name, group in df.groupby('gcm', sort=False):
        group = group.copy()

        # 1. Identify drought months (handles both boolean and 1/0 representation)
        is_dry = (group[dry_col] == 1) | (group[dry_col] == True)

        # 2. Apply pooling strategy: Include a non-drought month if it is sandwiched between two drought months
        pooled_dry = is_dry | (is_dry.shift(1).fillna(False) & is_dry.shift(-1).fillna(False))

        if not pooled_dry.any():
            logger.warning(f'No drought events found for GCM: {gcm_name}')
            continue

        # 3. Define unique consecutive event IDs within this GCM
        # An event starts when a month is dry/pooled-dry, but the previous month was not
        event_start = pooled_dry & (~pooled_dry.shift(1).fillna(False))
        group['event_id'] = event_start.cumsum()

        # 4. Filter out non-drought records
        drought_records = group[pooled_dry]

        # 5. Aggregate metrics per event
        event_summary = drought_records.groupby('event_id').agg(
            year=('year', 'first'),
            start_month=('month', 'first'),
            end_month=('month', 'last'),
            event_len=(spei_col, 'count'),  # Total consecutive months inside the pooled event
            event_intensity=(spei_col, 'sum')  # Cumulative sum of SPEI3 inside the pooled event
        ).reset_index()

        # Add the GCM context column
        event_summary['gcm'] = gcm_name

        # Reorder columns to match specifications exactly
        event_summary = event_summary[[
            'year', 'start_month', 'end_month', 'event_id', 'gcm', 'event_len', 'event_intensity'
        ]]

        all_gcm_events.append(event_summary)

    # Combine results from all GCMs into a single DataFrame
    if all_gcm_events:
        return pd.concat(all_gcm_events, ignore_index=True)
    else:
        return pd.DataFrame(
            columns=['year', 'start_month', 'end_month', 'event_id', 'gcm', 'event_len', 'event_intensity'])



def events_count_main(config):
    logger.info("Starting Stage 3: Drought Events Lenght and Duration Extraction")
    
    spei_csv = config.get("spei_csv")
    if not spei_csv or not os.path.exists(spei_csv):
        logger.error(f"Input file not found: {spei_csv}")
        sys.exit(1)
        
    logger.info(f"Loading SPEI calculated datasets from: {spei_csv}")
    df = pd.read_csv(spei_csv)
    
    # Load Basic Configuration Params
    su_col = config.get("spatial_unit_col", "region")
    spei_indices = config.get("spei_indices", [3, 12])
    spei_3_months = config.get("spei_3_months", [5, 6, 7, 8])
    
    # Load Expected Output File Templates
    dry_events_csv_tmpl = config.get("dry_events_csv")
    dry_events_len_csv_tmpl = config.get("dry_events_len_csv")
    dry_events_sev_csv_tmpl = config.get("dry_events_sev_csv")
    
    for scale in spei_indices:
        spei_col = f"spei{scale}"
        
        logger.info(f"Processing Events for Index: {spei_col}")
        if spei_col not in df.columns:
            logger.warning(f"{spei_col} column not found in input data. Skipping.")
            continue

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
