

import os
from ms_performance_prechecker.prechecker.register import RrecheckerBase
from ms_performance_prechecker.prechecker.utils import str_ignore_case, logger, set_log_level, deep_compare_dict
from ms_performance_prechecker.prechecker.utils import MIES_INSTALL_PATH, MINDIE_SERVICE_DEFAULT_PATH
from ms_performance_prechecker.prechecker.utils import read_csv_or_json


class MindieConfigCollecter(RrecheckerBase):
    __checker_name__ = "MindieConfig"
    def collect_env(self, **kwargs):
        mindie_service_path =  kwargs.get("mindie_service_path")
        
        if mindie_service_path is None:
            mindie_service_path = os.getenv(MIES_INSTALL_PATH, MINDIE_SERVICE_DEFAULT_PATH)
        if not os.path.exists(mindie_service_path):
            logger.warning(f"mindie config.json: {mindie_service_path} not exists, will skip related checkers")
            return None

        mindie_service_config = read_csv_or_json(os.path.join(mindie_service_path, "conf", "config.json"))
        logger.debug(
            f"mindie_service_config: {get_next_dict_item(mindie_service_config) if mindie_service_config else None}"
        )
        return mindie_service_config
    
mindie_config_collecter = MindieConfigCollecter()