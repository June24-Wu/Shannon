import socket
import time
from config.config import TEST_FACTORS_DAILY_TI0, FACTORS_PATH, FACTOR_START_DATE
from Research.utils.namespace import load_namespace
from feature.ops import Distribute
from feature.ops import FeatureOps
from Research.utils import normalization as norm
import os

start_date = FACTOR_START_DATE
end_date = '2021-07-01'

config_path = r'/config/feature_bt_template_test'
print('Loading the configuration from ' + config_path)
configs = load_namespace(config_path)
FT = FeatureOps(configs, TEST_FACTORS_DAILY_TI0, logger=True,
                check_table=True, check_corr=False, feature_path=FACTORS_PATH)
output_folder = os.path.join(configs.corr_path, f"{configs.db}", 'features')
dist = Distribute(configs)

if FT.logger is not None:
    FT.logger.info("=" * 125)
    FT.logger.info(f"Task of feature test starts, worker: {socket.gethostname()}...")

while True:
    result = dist.get_one_factor_from_redis_for_file()
    if result is None:
        time.sleep(10)
        continue
    researcher = result['uname']
    alpha_name = result['ft_name']
    alpha_in_db = FT.features_in_db

    if alpha_name in alpha_in_db:
        FT.logger.warning(f"Feature {alpha_name} already in feature db, no action...")

    else:
        # try:
        if FT.logger is not None:
            FT.logger.info("=" * 125)
            FT.logger.info(f"Test of {alpha_name} starts...")
        FT.distribute_upload_feature_from_file(researcher, alpha_name, start_date, end_date,
                                               universe='Investable',
                                               timedelta=None, transformer=norm.standard_scale,
                                               output=output_folder, verbose=True)
        print('ok')

        # except Exception as err:
        #     FT.reset()
        #     if FT.logger is not None:
        #         FT.logger.error(f"Test of {alpha_name} fails, error message: {err}")
        #     continue
