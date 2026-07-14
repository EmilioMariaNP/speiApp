import pandas as pd
import numpy as np
import os
import sys
from loguru import logger

# Ensure project root is in python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.utils import load_config, save_dataframe, generate_ai_commentary

# def compute_drought_events(df, dry_col = 'spei3_dry', spei_col = 'spei3'):
#     """
#     Computes the length and intensity of drought events for each GCM
#     using a single-month pooling strategy.
#     """
#     # Ensure the dataframe is chronologically sorted per GCM
#     df = df.sort_values(by=['gcm', 'year', 'month']).reset_index(drop=True)
#
#     all_gcm_events = []
#
#     # Process each GCM group independently
#     for gcm_name, group in df.groupby('gcm', sort=False):
#         logger.info(f'Processing GCM: {gcm_name}')
#         group = group.copy()
#
#         # 1. Identify drought months (handles both boolean and 1/0 representation)
#         is_dry = (group[dry_col] == 1) | (group[dry_col] == True)
#
#         # 2. Apply pooling strategy: Include a non-drought month if it is sandwiched between two drought months
#         pooled_dry = is_dry | (is_dry.shift(1, fill_value=False) & is_dry.shift(-1, fill_value=False))
#
#         if not pooled_dry.any():
#             continue
#
#         # 3. Define unique event IDs
#         # FIX: Remove .fillna(False) here as well, and use fill_value=False inside shift()
#         event_start = pooled_dry & (~pooled_dry.shift(1, fill_value=False))
#         group['event_id'] = event_start.cumsum()
#
#         # 4. Filter out non-drought records
#         drought_records = group[pooled_dry]
#
#         # 5. Aggregate metrics per event
#         event_summary = drought_records.groupby('event_id').agg(
#             year=('year', 'first'),
#             start_month=('month', 'first'),
#             end_month=('month', 'last'),
#             event_len=(spei_col, 'count'),  # Total consecutive months inside the pooled event
#             event_intensity=(spei_col, 'sum')  # Cumulative sum of SPEI3 inside the pooled event
#         ).reset_index()
#
#         # Add the GCM context column
#         event_summary['gcm'] = gcm_name
#
#         # Reorder columns to match specifications exactly
#         event_summary = event_summary[[
#             'year', 'start_month', 'end_month', 'event_id', 'gcm', 'event_len', 'event_intensity'
#         ]]
#
#         all_gcm_events.append(event_summary)
#
#     # Combine results from all GCMs into a single DataFrame
#     if all_gcm_events:
#         return pd.concat(all_gcm_events, ignore_index=True)
#     else:
#         return pd.DataFrame(
#             columns=['year', 'start_month', 'end_month', 'event_id', 'gcm', 'event_len', 'event_intensity'])



def calculate_drought_pooling_events(df, spei_col, pooling_threshold=1, min_duration=2):
    """
        Identifies drought events using Runs Theory with an Inter-event Time pooling criterion.

        Args:
            df: DataFrame with 'is_drought' (1/0) and SPEI values.
            spei_col: The column name for SPEI.
            pooling_threshold: Max number of months between events to merge them (e.g., 1).
            min_duration: Minimum duration to keep an event after pooling.
        """
    if df.empty:
        return pd.DataFrame()

    # Ensure index is clean for distance calculation
    df = df.reset_index(drop=True)

    # 1. Identify start and end indices of raw drought sequences (is_drought == 1)
    is_drought = df['is_drought'].values
    padded = np.pad(is_drought, (1, 1), 'constant', constant_values=0)
    diff = np.diff(padded)

    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0] - 1  # Inclusive ends

    if len(starts) == 0:
        return pd.DataFrame()

    # Create initial list of [start_idx, end_idx]
    raw_events = list(zip(starts, ends))

    # 2. Apply Pooling Logic
    pooled_events = []
    if raw_events:
        curr_start, curr_end = raw_events[0]

        for i in range(1, len(raw_events)):
            next_start, next_end = raw_events[i]
            # Gap is the number of non-drought months between events
            # Example: end at index 3, start at index 5 -> gap is 1 month (index 4)
            gap_months = next_start - curr_end - 1

            if gap_months <= pooling_threshold:
                # Merge: Extend the current event to the end of the next one
                curr_end = next_end
            else:
                # Close current event and start a new one
                pooled_events.append((curr_start, curr_end))
                curr_start, curr_end = next_start, next_end

        pooled_events.append((curr_start, curr_end))

    # 3. Aggregate results using the pooled indices
    results = []
    for s, e in pooled_events:
        duration = e - s + 1
        # Intensity is the sum of SPEI over the whole span (including the gap)
        # We use .iloc[s:e+1] because e is inclusive
        total_intensity = df.loc[s:e, spei_col].sum()

        results.append({
            'start_year': df.loc[s, 'year'],
            'start_month': df.loc[s, 'month'],
            'end_month': df.loc[e, 'month'],
            'duration': duration,
            'intensity': round(total_intensity * -1, 2)  # Convert to positive magnitude
        })

    df_results = pd.DataFrame(results)

    # 4. Filter by minimum duration (removes short/noisy events)
    if not df_results.empty:
        df_results = df_results[df_results['duration'] >= min_duration]

    return df_results



def events_count_main(config):
    logger.info("Starting Stage 3: Drought Events Length and Duration Extraction")

    spei_csv = config.get("spei_csv")
    if not spei_csv or not os.path.exists(spei_csv):
        logger.error(f"Input file not found: {spei_csv}")
        sys.exit(1)

    logger.info(f"Loading SPEI calculated datasets from: {spei_csv}")
    df_spei = pd.read_csv(spei_csv)

    # Load Basic Configuration Params
    su_col = config.get("spatial_unit_col", "region")
    spei_indices = config.get("spei_indices", [3, 12])
    spei_threshold = config.get('spei_threshold', -1)

    scenarios = df_spei['scenario'].unique().tolist()

    # limit analysis to selected spatial units, if provided
    if len(config.get('spatial_units', []))> 0:
        df_spei = df_spei[df_spei[su_col].isin(config['spatial_units'])]

    spatial_units = df_spei[su_col].unique().tolist()
    gcms = df_spei['gcm'].unique().tolist()

    # Load Expected Output File Templates
    dry_events_csv = config.get("dry_events_csv")
    dry_events_count_csv = config.get('dry_events_count_csv')

    # dfs = []
    # for i, spei_col in enumerate(spei_cols):
    #     spei_dry_col = spei_dry_cols[i]
    #
    #     #logger.info(f"Processing Events for Index: {spei_col}")
    #     if spei_col not in df.columns:
    #         logger.warning(f"{spei_dry_col} column not found in input data. Skipping.")
    #         continue
    #
    #     scale_df = df.copy()
    #
    #     for su in spatial_units:
    #         for scenario in scenarios:
    #             logger.info(f'Processing {su}-{scenario}-{spei_col}')
    #             df_su_scen = scale_df[(scale_df['scenario'] == scenario) & (scale_df[su_col] == su)]
    #             # Extract Distinct Drought Event characteristics (Duration, Severity)
    #             events_df = compute_drought_events(df = df_su_scen,
    #                                                dry_col = spei_dry_col,
    #                                                spei_col = spei_col
    #                                                )
    #             events_df[su_col] = su
    #             events_df['scenario'] = scenario
    #             events_df['spei'] = spei_col
    #            dfs.append(events_df)

    df_events_list = []
    for su in spatial_units:
        for spei_scale in spei_indices:
            spei_col = f'spei{spei_scale}'

            for scenario in scenarios:
                for gcm in gcms:
                    logger.info(f'Evaluating {scenario}-{gcm}-{spei_col} for {su}')

                    df_select = df_spei[['year', 'month', su_col, 'scenario', 'gcm', spei_col]]

                    df_select = df_select[(df_select[su_col] == su) &
                                          (df_select['gcm'] == gcm) &
                                          (df_select['scenario'] == scenario)
                                          ]
                    mask = df_select[spei_col] <= spei_threshold
                    df_select['is_drought'] = np.where(mask, 1, 0)

                    # if spei_scale == 3:
                    #     df_select = df_select[df_select['month'].isin(growth_season_months)]

                    # drought_events = calculate_drought_events(df_select, spei_col)
                    drought_events = calculate_drought_pooling_events(df=df_select,
                                                                      spei_col=spei_col,
                                                                      pooling_threshold=1,
                                                                      min_duration=2)

                    drought_events['scenario'] = scenario
                    drought_events['gcm'] = gcm
                    drought_events['spei'] = spei_scale
                    drought_events[su_col] = su
                    df_events_list.append(drought_events)

    # Ensure structural integrity corresponding to filepaths
    os.makedirs(os.path.dirname(dry_events_csv), exist_ok=True)

    # Export processed frames cleanly
    df_events = pd.concat(df_events_list, ignore_index=True)
    df_events.to_csv(dry_events_csv, index=False)
    logger.info(f"Saved total counts CSV format to -> {dry_events_csv}")

    df_count = df_events.groupby(by=[su_col, 'scenario', 'gcm', 'spei'])['duration'].count().reset_index()
    df_count.rename(columns={'duration': 'events_count'}, inplace=True)
    save_dataframe(df_count, dry_events_count_csv)



if __name__ == "__main__":

    # Handle the JSON path cleanly whether executing locally from the /processing dir or globally
    # config_path = '../data/config_nut2.json'
    config_path = '../data/config_ecoregions.json'
    if not os.path.exists(config_path):
        config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), config_path)

    if not os.path.exists(config_path):
        logger.error(f"Cannot discover JSON setting path securely using fallback root at {config_path}")
        sys.exit(1)

    config = load_config(config_path)


    # update file paths
    home_dir = config.get('home_dir', None)
    if home_dir:
        config['dry_events_csv'] = os.path.join(home_dir, config['dry_events_csv'])
        config['spei_csv'] = os.path.join(home_dir, config['spei_csv'])
        config['dry_events_count_csv'] = os.path.join(home_dir, config['dry_events_count_csv'])


    events_count_main(config)
    logger.info("Stage 3 completed effectively.")
