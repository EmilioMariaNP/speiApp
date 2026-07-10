import os
from loguru import logger

from processing.spei_calc import spei_calc_main
from processing.gcms_eval import main_gcms_eval
from utils.utils import load_config, load_dataframe, dump_json_to_file

if __name__ == "__main__":
    # Standard entrypoint loading the generic config json
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'config.json')
    config_dict = load_config(config_path)
    if config_dict:

        if config_dict.get("execute_spei", False):
            spei_calc_main(config_dict)

        try:


            if config_dict.get('execute_ks_test', False):
                config_dict = main_gcms_eval(config_dict)
                dump_json_to_file(config_dict, config_path, indent=4)


        except Exception as e:
            logger.error(e)



    else:
        logger.error("Failed to load config.json. Exiting.")

    print('Process completed.')

