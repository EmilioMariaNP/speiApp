library(SPEI)
library(readr)
library(dplyr)

calculate_spei <- function(
  simulation_dataset_csv,
  validation_dataset_csv,
  spei_scales,
  validation_label,
  out_csv,
  ref_start_year,
  ref_end_year,
  spatial_unit_column = 'region',
  water_balance_col = 'water_balance',
  reference_scenario = 'historical'
) {
  # 1. Read files
  message(paste("Reading simulation dataset:", simulation_dataset_csv))
  df1 <- readr::read_csv(simulation_dataset_csv, show_col_types = FALSE)
  
  message(paste("Reading validation dataset:", validation_dataset_csv))
  df2 <- readr::read_csv(validation_dataset_csv, show_col_types = FALSE)
  
  # 2. Auto-detect if swapped
  is_df1_sim <- reference_scenario %in% unique(df1$scenario) && length(unique(df1$scenario)) > 1
  is_df2_sim <- reference_scenario %in% unique(df2$scenario) && length(unique(df2$scenario)) > 1
  
  if (is_df2_sim && !is_df1_sim) {
    message("Detected swapped arguments. Internally swapping datasets so simulation_dataset_csv contains simulated data and validation_dataset_csv contains validation data.")
    sim_df <- df2
    val_df <- df1
  } else {
    sim_df <- df1
    val_df <- df2
  }
  
  # Filter simulated dataset to keep only years >= ref_start_year
  sim_df <- sim_df %>% filter(year >= ref_start_year)

  
  # 3. Standardize spatial unit column to 'region'
  if (spatial_unit_column != "region") {
    if (spatial_unit_column %in% colnames(sim_df)) {
      sim_df <- sim_df %>% rename(region = !!sym(spatial_unit_column))
    }
    if (spatial_unit_column %in% colnames(val_df)) {
      val_df <- val_df %>% rename(region = !!sym(spatial_unit_column))
    }
  }
  
  # Ensure essential columns exist
  if (!"region" %in% colnames(sim_df)) {
    stop(paste("Spatial unit column '", spatial_unit_column, "' not found in simulation dataset."))
  }
  if (!"region" %in% colnames(val_df)) {
    stop(paste("Spatial unit column '", spatial_unit_column, "' not found in validation dataset."))
  }
  # if (!spatial_unit_column %in% colnames(sim_df)) {
  #   stop(paste("Spatial unit column '", spatial_unit_column, "' not found in simulation dataset."))
  # }
  # if (!spatial_unit_column %in% colnames(val_df)) {
  #   stop(paste("Spatial unit column '", spatial_unit_column, "' not found in validation dataset."))
  # }
  if (!water_balance_col %in% colnames(sim_df)) {
    stop(paste("Water balance column '", water_balance_col, "' not found in simulation dataset."))
  }
  if (!water_balance_col %in% colnames(val_df)) {
    stop(paste("Water balance column '", water_balance_col, "' not found in validation dataset."))
  }
  
  # Ensure columns are typed correctly
  sim_df$region <- as.character(sim_df$region)
  val_df$region <- as.character(val_df$region)
  
  # 4. Process validation dataset (group by region)
  message("Processing validation dataset...")
  val_results <- list()
  unique_val_regions <- unique(val_df$region)
  
  for (reg in unique_val_regions) {
    reg_df <- val_df %>% 
      filter(region == reg) %>% 
      arrange(year, month)
    
    if (nrow(reg_df) == 0) next
    
    val_min_year <- min(reg_df$year)
    val_min_month <- reg_df$month[which.min(reg_df$year * 12 + reg_df$month)]
    val_max_year <- max(reg_df$year)
    val_max_month <- reg_df$month[which.max(reg_df$year * 12 + reg_df$month)]
    
    # Adjust ref.start and ref.end to be within the data range
    ref_start <- c(ref_start_year, 1)
    if (ref_start_year < val_min_year || (ref_start_year == val_min_year && 1 < val_min_month)) {
      ref_start <- c(val_min_year, val_min_month)
    }
    ref_end <- c(ref_end_year, 12)
    if (ref_end_year > val_max_year || (ref_end_year == val_max_year && 12 > val_max_month)) {
      ref_end <- c(val_max_year, val_max_month)
    }
    
    # Create ts object
    ts_data <- ts(reg_df[[water_balance_col]], frequency = 12, start = c(val_min_year, val_min_month))
    
    for (scale in spei_scales) {
      if ((ref_start[1] * 12 + ref_start[2]) > (ref_end[1] * 12 + ref_end[2])) {
        warning(paste("Reference period outside data range for validation region", reg, "- assigning NA"))
        reg_df[[paste0("spei", scale)]] <- NA
      } else {
        spei_res <- spei(ts_data, scale = scale, ref.start = ref_start, ref.end = ref_end)
        reg_df[[paste0("spei", scale)]] <- as.numeric(spei_res$fitted)
      }
    }
    
    # Enforce scenario = 'validation' and gcm = validation_label
    reg_df$scenario <- "validation"
    reg_df$gcm <- validation_label
    
    val_results[[length(val_results) + 1]] <- reg_df
  }
  
  val_out_df <- bind_rows(val_results)
  
  # 5. Process simulation dataset
  message("Processing simulation dataset...")
  sim_results <- list()
  unique_sim_regions <- unique(sim_df$region)
  unique_sim_gcms <- unique(sim_df$gcm)
  unique_sim_scenarios <- unique(sim_df$scenario)
  
  ssp_scenarios <- unique_sim_scenarios[unique_sim_scenarios != reference_scenario]
  
  for (reg in unique_sim_regions) {
    for (g in unique_sim_gcms) {
      # Extract historical data for this region and GCM
      hist_sub <- sim_df %>% 
        filter(region == reg, gcm == g, scenario == reference_scenario) %>% 
        arrange(year, month)
      
      for (ssp in ssp_scenarios) {
        ssp_sub <- sim_df %>% 
          filter(region == reg, gcm == g, scenario == ssp) %>% 
          arrange(year, month)
        
        if (nrow(hist_sub) == 0 && nrow(ssp_sub) == 0) next
        
        # Combine historical and SSP to construct the reference baseline and projection series
        combined_sub <- bind_rows(hist_sub, ssp_sub) %>% arrange(year, month)
        
        comb_min_year <- combined_sub$year[1]
        comb_min_month <- combined_sub$month[1]
        
        ts_data <- ts(combined_sub[[water_balance_col]], frequency = 12, start = c(comb_min_year, comb_min_month))
        
        for (scale in spei_scales) {
          # Safeguard ref.start / ref.end against data boundaries
          comb_max_year <- max(combined_sub$year)
          comb_max_month <- combined_sub$month[which.max(combined_sub$year * 12 + combined_sub$month)]
          
          ref_start <- c(ref_start_year, 1)
          if (ref_start_year < comb_min_year || (ref_start_year == comb_min_year && 1 < comb_min_month)) {
            ref_start <- c(comb_min_year, comb_min_month)
          }
          ref_end <- c(ref_end_year, 12)
          if (ref_end_year > comb_max_year || (ref_end_year == comb_max_year && 12 > comb_max_month)) {
            ref_end <- c(comb_max_year, comb_max_month)
          }
          
          if ((ref_start[1] * 12 + ref_start[2]) > (ref_end[1] * 12 + ref_end[2])) {
            warning(paste("Reference period outside data range for region", reg, "GCM", g, "SSP", ssp, "- assigning NA"))
            combined_sub[[paste0("spei", scale)]] <- NA
          } else {
            spei_res <- spei(ts_data, scale = scale, ref.start = ref_start, ref.end = ref_end)
            combined_sub[[paste0("spei", scale)]] <- as.numeric(spei_res$fitted)
          }
        }
        
        sim_results[[length(sim_results) + 1]] <- combined_sub
      }
    }
  }
  
  sim_out_df <- bind_rows(sim_results)
  
  # Remove duplicates in historical rows generated from binding historical with multiple SSPs
  sim_out_df <- sim_out_df %>% 
    distinct(year, month, region, scenario, gcm, .keep_all = TRUE)
  
  # 6. Combine, add date in format DD/MM/YY, and select required columns
  final_df <- bind_rows(val_out_df, sim_out_df)
  
  final_df$date <- format(as.Date(paste(final_df$year, final_df$month, "01", sep="-"), format="%Y-%m-%d"), "%d/%m/%Y")
  
  required_cols <- c("date", "year", "month", "region", "scenario", "gcm", paste0("spei", spei_scales))
  
  final_df <- final_df %>% 
    select(all_of(required_cols))

  # renme the spatial unit column to its original name
  if (spatial_unit_column != "region" && "region" %in% colnames(final_df)) {
    final_df <- final_df %>%
      rename(!!spatial_unit_column := region)
  }

  
  # 7. Write output to CSV
  out_dir <- dirname(out_csv)
  if (!dir.exists(out_dir)) {
    dir.create(out_dir, recursive = TRUE)
  }
  
  readr::write_csv(final_df, out_csv)
  message(paste("SPEI calculations successfully written to:", out_csv))
  
  return(final_df)
}

calculate_spei_from_json <- function(config_file) {
  library(jsonlite)
  
  config_list <- fromJSON(config_file)


  home_dir <- config_list$home_dir

  water_balance_col<- config_list$water_balance_col
  spatial_unit_column <- config_list$spatial_unit_col
  reference_scenario <- config_list$reference_scenario
  
  simulation_dataset_csv <- config_list$projections_csv
  validation_dataset_csv <- config_list$validation_dataset
  
  ref_start_year<- config_list$ref_start_year
  ref_end_year<- config_list$ref_end_year
  
  spei_scales <- config_list$spei_indices
  validation_label <- 'cru'
  
  out_csv <- config_list$spei_csv

    if (!is.null(home_dir)) {

      simulation_dataset_csv <- file.path(home_dir, simulation_dataset_csv)
      validation_dataset_csv <- file.path(home_dir, validation_dataset_csv)
      out_csv <- file.path(home_dir, out_csv)
  }
  
  # If out_csv is a relative filename, resolve it using the home_dir configuration
  if (!grepl("^/", out_csv) && !is.null(config_list$home_dir)) {
    out_csv <- file.path(config_list$home_dir, out_csv)
  }
  
  df_res <- calculate_spei(
    simulation_dataset_csv = simulation_dataset_csv,
    validation_dataset_csv = validation_dataset_csv,
    spei_scales = spei_scales,
    validation_label = validation_label,
    out_csv = out_csv,
    ref_start_year = ref_start_year,
    ref_end_year = ref_end_year,
    spatial_unit_column = spatial_unit_column,
    water_balance_col = water_balance_col,
    reference_scenario = reference_scenario
  )
  
  return(df_res)
}
if (!interactive()) {
  args <- commandArgs(trailingOnly = TRUE)
  if (length(args) > 0) {
    config_file <- args[1]
    calculate_spei_from_json(config_file)
  }
}

config_file <- '/home/politti/git/speiApp/data/config_nut2_1_min.json'
calculate_spei_from_json(config_file)
