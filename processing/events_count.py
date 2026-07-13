import pandas as pd
import numpy as np
import os
import sys
from loguru import logger

# Ensure project root is in python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.utils import load_config, save_dataframe, generate_ai_commentary

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
        logger.info(f'Processing GCM: {gcm_name}')
        group = group.copy()

        # 1. Identify drought months (handles both boolean and 1/0 representation)
        is_dry = (group[dry_col] == 1) | (group[dry_col] == True)

        # 2. Apply pooling strategy: Include a non-drought month if it is sandwiched between two drought months
        pooled_dry = is_dry | (is_dry.shift(1, fill_value=False) & is_dry.shift(-1, fill_value=False))

        if not pooled_dry.any():
            continue

        # 3. Define unique event IDs
        # FIX: Remove .fillna(False) here as well, and use fill_value=False inside shift()
        event_start = pooled_dry & (~pooled_dry.shift(1, fill_value=False))
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
    logger.info("Starting Stage 3: Drought Events Length and Duration Extraction")

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
    spei_cols = [f'spei{x}' for x in spei_indices]
    spei_dry_cols = [f'spei{x}_dry' for x in spei_indices]

    scenarios = df['scenario'].unique().tolist()
    spatial_units = df[su_col].unique().tolist()

    # Load Expected Output File Templates
    dry_events_csv = config.get("dry_events_csv")
    dry_events_count_csv = config.get('dry_events_count_csv')

    dfs = []
    for i, spei_col in enumerate(spei_cols):
        spei_dry_col = spei_dry_cols[i]

        #logger.info(f"Processing Events for Index: {spei_col}")
        if spei_col not in df.columns:
            logger.warning(f"{spei_dry_col} column not found in input data. Skipping.")
            continue

        scale_df = df.copy()

        # 2. For Agricultural scales (spei3), limit months explicitly
        if spei_col == 'spei3':
            logger.info(f"Limiting agricultural scale 3 to specified months: {spei_3_months}")
            scale_df = scale_df[scale_df['month'].isin(spei_3_months)]


        for su in spatial_units:
            for scenario in scenarios:
                logger.info(f'Processing {su}-{scenario}-{spei_col}')
                df_su_scen = scale_df[(scale_df['scenario'] == scenario) & (scale_df[su_col] == su)]
                # Extract Distinct Drought Event characteristics (Duration, Severity)
                events_df = compute_drought_events(df = df_su_scen,
                                                   dry_col = spei_dry_col,
                                                   spei_col = spei_col
                                                   )
                events_df[su_col] = su
                events_df['scenario'] = scenario
                events_df['spei'] = spei_col

                dfs.append(events_df)


    # Ensure structural integrity corresponding to filepaths
    os.makedirs(os.path.dirname(dry_events_csv), exist_ok=True)

    # Export processed frames cleanly
    df_events = pd.concat(dfs, ignore_index=True)
    df_events.to_csv(dry_events_csv, index=False)
    logger.info(f"Saved total counts CSV format to -> {dry_events_csv}")

    df_count = df_events.groupby(by=[su_col, 'scenario', 'gcm', 'spei', 'event_id']).count().reset_index()
    save_dataframe(df_count, dry_events_count_csv)



if __name__ == "__main__":

    # Handle the JSON path cleanly whether executing locally from the /processing dir or globally
    config_path = '../data/config_nut2.json'
    if not os.path.exists(config_path):
        config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data/config_nut2.json")

    if not os.path.exists(config_path):
        logger.error(f"Cannot discover JSON setting path securely using fallback root at {config_path}")
        sys.exit(1)

    config = load_config(config_path)

    events_count_main(config)
    logger.info("Stage 3 completed effectively.")
