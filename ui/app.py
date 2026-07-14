import os
import subprocess
from loguru import logger
import pandas as pd

from processing.events_count import events_count_main
from processing.gcms_eval import main_gcms_eval
from utils.utils import load_config, load_dataframe, dump_json_to_file


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
    # Standard entrypoint loading the generic config json
    # config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'config_nut2.json')
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data",
        "config_ecoregions.json",
    )
    config_dict = load_config(config_path)

    # update file paths
    home_dir = config_dict.get("home_dir", None)
    if home_dir:
        config_dict["projections_csv"] = os.path.join(
            home_dir, config_dict["projections_csv"]
        )
        config_dict["validation_dataset"] = os.path.join(
            home_dir, config_dict["validation_dataset"]
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

    if config_dict:

        if config_dict.get("execute_spei", False):
            run_spei_calc_r(config_path)
            # fix NA values from R
        try:
            if config_dict.get("execute_ks_test", False):
                config_dict = main_gcms_eval(config_dict)
                #dump_json_to_file(config_dict, config_path, indent=4)
            if config_dict.get("execute_count", False):
                events_count_main(config_dict)

        except Exception as e:
            logger.error(e)

    else:
        logger.error("Failed to load config_nut2.json. Exiting.")

    print("Process completed.")
