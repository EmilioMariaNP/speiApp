import os
from loguru import logger

from processing.events_count import events_count_main
from processing.spei_calc import spei_calc_main
from processing.gcms_eval import main_gcms_eval
from utils.utils import load_config, load_dataframe, dump_json_to_file

if __name__ == "__main__":
    # Standard entrypoint loading the generic config json
    # config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'config_nut2.json')
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'config_ecoregions.json')
    config_dict = load_config(config_path)

    # update file paths
    home_dir = config_dict['home_dir']
    config_dict['projections_csv'] = os.path.join(home_dir, config_dict['projections_csv'])
    config_dict['validation_dataset'] = os.path.join(home_dir, config_dict['validation_dataset'])
    print(config_dict['validation_dataset'])
    config_dict['dry_events_count_csv'] = os.path.join(home_dir, config_dict['dry_events_count_csv'])

    if config_dict:

        if config_dict.get("execute_spei", False):
            spei_calc_main(config_dict)

        try:
            if config_dict.get('execute_ks_test', False):
                config_dict = main_gcms_eval(config_dict)
                dump_json_to_file(config_dict, config_path, indent=4)
            if config_dict.get('execute_count', False):
                events_count_main(config_dict)


        except Exception as e:
            logger.error(e)



    else:
        logger.error("Failed to load config_nut2.json. Exiting.")

    print('Process completed.')

