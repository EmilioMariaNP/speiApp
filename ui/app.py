import os
import subprocess
from loguru import logger
import sys

from processing.copula_aggregation import main_copula_aggregation
from processing.copula_analysis_nn_station import main_copula_nn_station
from processing.events_count import events_count_main
from processing.gcms_eval import main_gcms_eval
from utils.utils import load_config


def run_spei_calc_r(config_path):
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    r_script_path = os.path.join(project_root, "processing", "spei_calc.R")
    logger.info(
        f"Invoking R script for SPEI calculation: {r_script_path} with config: {config_path}"
    )
    try:
        _result = subprocess.run(
            ["Rscript", r_script_path, config_path],
            capture_output=True,
            text=True,
            check=True,
        )
        # logger.info("R script output:")
        # print(result.stdout)
    except subprocess.CalledProcessError as e:
        logger.error(f"R script failed with error (exit code {e.returncode}):")
        print(e.stderr)
        raise e


if __name__ == "__main__":

    #config_file_name = "config_ecoregions.json"
    config_file_name = 'config_nut2.json'

    
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    else:
        # Standard entrypoint loading the generic config json
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data",
            config_file_name
        )
    config_dict = load_config(config_path)

    # update file paths
    home_dir = config_dict.get("home_dir", None)
    if home_dir:
        config_dict["projections_csv"] = os.path.join(
            home_dir, config_dict["projections_csv"]
        )
        config_dict["validation_dataset_csv"] = os.path.join(
            home_dir, config_dict["validation_dataset_csv"]
        )
        config_dict["dry_events_count_csv"] = os.path.join(
            home_dir, config_dict["dry_events_count_csv"]
        )
        config_dict["dry_events_csv"] = os.path.join(
            home_dir, config_dict["dry_events_csv"]
        )
        config_dict['spei_csv'] = os.path.join(home_dir, config_dict["spei_csv"])

        config_dict['gcm_eval_csv'] = os.path.join(home_dir, config_dict['gcm_eval_csv'])
        config_dict['copula_analysis_csv'] = os.path.join(home_dir, config_dict['copula_analysis_csv'])
        config_dict['copula_analysis_aggregation_csv'] = os.path.join(
            home_dir, config_dict['copula_analysis_aggregation_csv'])

        growth_season_start = int(config_dict.get('growth_season_start', 1))
        growth_season_end = int(config_dict.get('growth_season_end', 12))
        growth_season_months = list(range(growth_season_start, growth_season_end + 1))
        config_dict['spei_3_months'] = growth_season_months

    if config_dict:

        if config_dict.get("execute_spei", False):
            run_spei_calc_r(config_path)
        try:
            if config_dict.get("execute_ks_test", False):
                config_dict = main_gcms_eval(config_dict)
            if config_dict.get("execute_count", False):
                events_count_main(config_dict)
            if config_dict.get("execute_copula", False):
                main_copula_nn_station(config_dict)
            if config_dict.get('execute_copula_aggregation', False):
                main_copula_aggregation(config_dict)


        except Exception as e:
            logger.exception(e)

    else:
        logger.error("Failed to load config_nut2.json. Exiting.")

    print("Process completed.")
