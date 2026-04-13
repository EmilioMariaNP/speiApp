# Specification Document: SPEI Futures Drought Analysis Application

## 1. Overview
The **SPEI Futures** application is a Python-based tool designed to process climate data to compute SPEI (Standardized Precipitation Evapotranspiration Index) indices (e.g., SPEI-3, 6, 12). It classifies months into drought/non-drought periods, and compares historical reference frequencies, durations, and intensities of drought events against climate change projections.

## 2. General Requirements and Coding Standards
*   **Python Environment**: Create a dedicated Python environment for this application based around `pandas`.
*   **Path Management**: No hard-coded paths or variables. All file paths, constants, and parameters must be loaded from a central JSON configuration file.
*   **Reusability**: Encapsulate reusable logic (e.g., config loading, common math operations) in a shared `utils.py` module. 
*   **Libraries**: Use the latest versions of standard scientific Python packages (e.g., `pandas`, `numpy`, `scipy`).
*   **Logging**: Use `loguru` for all console output. Utilize `logger.info`, `logger.warning`, and `logger.error` appropriately. Avoid standard `print()` statements.
*   **Code Structure**: 
    *   Avoid long, monolithic scripts. Encapsulate execution into small, focused functions.
    *   Each executable module must have a primary entry point named after the module followed by `_main` (e.g., `spei_calc_main(config)`).
    *   Include an `if __name__ == "__main__":` block in each execution script to load the JSON config into a dictionary and pass it to the main function.
*   **Documentation**: Write code documentation (docstrings) and produce external documentation under the `Documentation/code` directory. Additionally, inside the `Documentation` folder, create a plain language summary of the execution stages explaining their objective, the method applied, the output, and how to interpret it.
*   **Spatial Unity Autonomy**: All stages must process data independently for each spatial unit identified in the input dataset column specified by the `spatial_unit_col` configuration key.
*   **Results Commentary**: At the end of each stage, the execution must generate an AI-written commentary explaining the results. Save the commentary in the same output folder as the stage's outputs. Name the file with the specific stage name followed by the suffix `_results_ai_comments`.

## 3. Project Architecture

### 3.1 Directory Structure
```text
Project_Root/
├── processing/                 # Core processing modules
│   ├── spei_calc.py            # Stage 1: Index Calculation
│   ├── gcms_eval.py            # Stage 2: Model Consistency
│   ├── events_count.py         # Stage 3: Event Counting
│   ├── copula_analysis.py      # Stage 4: Joint Probabilities
│   └── plots.py                # Plotting functional repository
├── utils/                      # Helper & Library functions
│   ├── __init__.py
│   ├── utils.py                # Generic utilities (Json loader, etc.)
│   └── copula_lib.py           # Copula mathematics
├── ui/                         # User Interface components
│   └── app.py                  # Platform independent UI 
├── Outputs/                    # Generated files & reports
│   ├── drought_counts/         # Outputs from Stage 3
│   ├── validation/
│   │   └── plots/
│   └── copula_analysis/
│       └── plots/
└── Documentation/
    ├── code/
    └── data/
```

### 3.2 Configuration JSON Structure
```json
{
  "spei_indices": ["spei3", "spei6", "spei12"],
  "reference_scenario": "reference",
  "spei_3_months": [5, 6, 7, 8],
  "projections_csv": "Inputs/projections_data.csv",
  "validation_dataset": "Inputs/validation_data.csv",
  "spatial_unit_col": "spatial_unit",
  "spatial_units": [],
  "spei_csv": "Outputs/spei_data.csv",
  "dry_events_csv": "Outputs/drought_counts/dry_events_{index}.csv",
  "dry_events_len_csv": "Outputs/drought_counts/dry_events_len_{index}.csv",
  "dry_events_sev_csv": "Outputs/drought_counts/dry_events_sev_{index}.csv",
  "gcm_eval_csv": "Outputs/gcm_evaluation.csv",
  "copula_analysis_csv": "Outputs/copula_analysis.csv",
  "spei_threshold": -0.5,
  "valid_gcms": {}
}
```

## 4. Processing Workflow

### Stage 1: SPEI Calculation (`processing/spei_calc.py`)
*   **Filtering**: Limit the analysis (across all stages) only to the spatial units specified in the JSON key `spatial_units`. If the JSON key `spatial_units` is `null` or the list is empty, perform the analysis for all available spatial units in the input dataset.
*   **Input Data**: Read external datasets using paths from `projections_csv` (modeled data) and `validation_dataset` (observed data). Both CSVs should contain `Year`, `Month`, `spatial unit`, `scenario`, `gcm`, `ETo`, `Pr`, `ETa`, and `water_balance`.
*   **Index Fitting Rule**: SPEI fitting utilizes the `water_balance` column. It must be computed independently for:
    1.  Observed Data (Empty GCM column).
    2.  Modeled GCM Data: For modeled data, fit the distribution *only* on the reference scenario for each specific GCM. Apply this fitted distribution locally to that specific GCM's climate projections.
*   **Output**: A Pandas dataframe saved as a CSV (path from `spei_csv`) containing all original columns plus the new calculated SPEI indices columns (e.g., `spei3`, `spei6`).

### Stage 2: Consistency Check (`processing/gcms_eval.py`)

**Objective:**
Before extracting drought events, validate the GCMs' ability to reliably reproduce historical SPEI statistics. Compare the GCM-simulated SPEI values during the reference period against the validation dataset to ensure statistical similarity in their distributions.

**Data Preparation & Filtering:**
1. Determine the validation dataset. Read the data corresponding to the validation dataset (its source is identified by the `validation_dataset` JSON key). If reading from the consolidated Stage 1 output, it is the data where `gcm` column values are empty. Normalize the scenario names by replacing them with `'validation'` where applicable.
2. Find the overlapping years between the validation dataset and the reference scenario (GCMs) by retrieving the name of the reference scenario directly from the JSON key `reference_scenario`.
3. Filter the data to only include this overlapping period for the comparison to ensure a fair evaluation.
4. Read the SPEI values calculated in **Stage 1** for each GCM and the validation dataset across each spatial unit and SPEI scale (e.g., spei3, spei12).

**Statistical Comparison (Kolmogorov-Smirnov Test):**
1. For each GCM, use the two-sample Kolmogorov-Smirnov (KS) test (`scipy.stats.ks_2samp`) to compare its SPEI distributions against the validation data.
2. Perform the KS test separately for each **SPEI scale**.
3. **Acceptance Criteria**: A GCM passes validation if the p-value is >= 0.05. If the p-value is < 0.05, the distributions are statistically different, and the GCM fails.
4. **Filtering Decision**: Apply a filtering approach where we record which GCM passes the test and which one does not for each SPEI scale individually.

**Outputs:**
All outputs should be saved to the directory `Outputs/validation`.

1. **Data Outputs:**
    - `ks_table.csv`: A comprehensive summary table containing all KS test statistics, p-values, and pass/fail statuses for every GCM, spatial unit, and SPEI scale.
    - `valid_gcms`: A dictionary where keys are SPEI scales (e.g., "spei3", "spei12") and values are lists of strings representing the GCMs that passed validation across all regions for that scale. This must be written in the config json and will be consumed by subsequent steps.
    - `validation.log`: An execution log capturing process steps, warnings, and final validation decisions.
2. **Visualizations:**
    - Generate grid plots comparing the Empirical Cumulative Distribution Functions (ECDFs) of the GCM SPEI values vs. the validation data SPEI values.
    - *Grid Structure*: Rows = Individual GCMs, Columns = spatial units.
    - *Content*: Overlay the ECDFs. Use **Black** for the validation data, **Blue** for the GCM if it *passed*, and **Red** for the GCM if it *failed*. Annotate each subplot with its corresponding p-value.
    - Create separate images for each SPEI scale: `Validation_Grid_spei3.png`, `Validation_Grid_spei12.png`, etc.

### Stage 3: Drought Events Extraction (`processing/events_count.py`)
*   **Execution Rule**: Runs separately for each SPEI index calculated in Stage 1. Groups data by `scenario` and `gcm`. For SPEI 3, consider only the months relevant for agriculture. Read these months from the JSON config using the key `spei_3_months`. Note: Only extract events for GCMs that passed the consistency check for the specific SPEI index being analyzed (as listed in the `valid_gcms` dictionary).
*   **Classification & Extraction Rules**: 
    1. Identify drought events using a threshold read from the JSON config key `spei_threshold`.
    2. Classify months as "dry" if `SPEI <= spei_threshold`.
*   **Calculations & Outputs**:
    Save all outputs of this stage inside the `Outputs/drought_counts/` subfolder.
    1.  **Count CSV**: Number of dry months per scenario-GCM and for the validation dataset. Path: `dry_events_csv` (dynamically modified to include the index name).
    2.  **Duration CSV**: Length of each distinct drought event (consecutive dry months below the threshold). Path: `dry_events_len_csv` (dynamically modified to include the index name).
    3.  **Severity CSV**: Cumulative absolute SPEI deficit during each distinct drought event. Path: `dry_events_sev_csv` (dynamically modified to include the index name).

### Stage 4: Copula Analysis (`processing/copula_analysis.py`)
*   **Objective**: Explore the joint probability of Frequency (severity) and Duration dynamically across reference and projected periods across Valid GCMs. 
*   **Execution Rule**: Must be performed first per scenario-GCM combination, and subsequently per scenario (Ensemble).
*   **Integration**: Utilizes `utils/copula_lib.py` for extraction, fitting Marginals, Copula selection (AIC evaluation of Clayton, Gumbel, Frank), and Return Period calculations.

## 5. Required Methodology & Tabular Outputs for Copula

### Methodology Constraints for Replication
1.  **Event Extraction**: Isolate periods where SPEI strictly falls below a defined baseline. Extract `Duration` and `Severity` (absolute sum).
2.  **Marginal Fitting**: Filter out `NaN`, `Inf`, and non-positive data before fitting Scipy `gamma` or `expon` distributions. Fall back to empirical CDFs if data is insufficient.
3.  **Copula Selection**: Transform marginal outputs to pseudo-observations (u,v). Evaluate Archimedean likelihoods using AIC to select the best dependency model.
4.  **RP Calculation**: Joint Return Period mathematically computed as: 
    RP = mu / 1-u-v+C(u,v)

### Tabular Outputs to Generate
Saved as separate CSVs under `Outputs/copula_analysis/`:

1.  **`copula_table.csv`** (Ensemble Copula Fits)
    *   *Columns*: Subasin, SPEI, Scenario, Type ("Ensemble"), Duration_Dist, Severity_Dist, Copula_Family, AIC/BIC, Avg_Interarrival.
2.  **`rp_table.csv`** (Ensemble Return Periods)
    *   *Columns*: Subasin, SPEI, Scenario, Univariate_Ref_RP, Ref_Threshold_Dur, Ref_Threshold_Sev, Computed_Joint_RP. 
    *   *Rule*: Thresholds must be calculated exclusively from the Ensemble's Reference Period.
3.  **`gcm_copula_table.csv`** (Per-GCM Copula Fits)
    *   Same as Table 1, but `Type` is replaced with `GCM`.
4.  **`gcm_rp_table.csv`** (Per-GCM Return Periods)
    *   Same as Table 2, but specific to individual GCM models. 
    *   *Critical Constraint*: `Ref_Threshold_Dur` and `Ref_Threshold_Sev` must be calculated locally from each GCM's independent Reference Period to measure relative intensification.

### Diagnostic Plots (`processing/plots.py`)
Saved individually for each spatial unit under `Outputs/copula_analysis/plots/`:
*   *Marginal Q-Q Plots/Histograms*: Goodness-of-fit for Gamma/Exponential.
*   *Pseudo-Observation Scatter Plots*: Comparing Copula dependency representation.
*   *Joint Return Period Contour Maps*: Non-linear progression of threshold probabilities.
*   *Trajectory Shift Plots*: Representing model consensus versus spread across GCM projections.

## 6. User Interface
*   **Module**: `ui/app.py`
*   **Functionality**: Platform-independent interface allowing a user to easily edit JSON configuration parameters, and select input/output file paths via native file-picker dialogs.
