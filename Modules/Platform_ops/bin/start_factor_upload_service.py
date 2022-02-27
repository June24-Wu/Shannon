import socket
import time
import traceback

from DevTools.tools.ding import Ding
from DevTools.tools.elk_logger import get_elk_logger
from Platform_ops.config.config import TEST_FACTORS_DAILY_TI0, configs, telephone_dict

from feature.ops import Distribute, FeatureOps
from utils.es_logger import getLogger

ding = Ding(configs.DING_SECRET_KEY, configs.DING_ACCESS_TOKEN)
FT = FeatureOps(configs, mysql_info=TEST_FACTORS_DAILY_TI0, logger=None,
                check_table=True, check_corr=False, feature_path=configs.FACTORS_PATH)
# output_folder = os.path.join(configs.corr_path, f"{configs.db}", 'features')
output_folder = None
logger = get_elk_logger("platform_correlation_service_test", console=True)
dist = Distribute(configs, logger=logger)

ding.send_ding("INFO | PLATFORM_UPLOAD_SERVICE",
               f"Task of feature upload starts, worker: {socket.gethostname()}...")
result_info = {"accepted": 1, "watched": 2, "rejected": 3}

while True:
    result = dist.get_one_factor_from_redis_for_dataframe()
    if result is None:
        time.sleep(10)
        continue

    researcher = result['uname']
    alpha_name = result['alpha_name']
    time_out = result['time_out']
    uuid = result['uuid']
    category = result['category']
    if category != 'PV':
        ding.send_ding("ERROR | PLATFORM_UPLOAD_SERVICE",
                       f"Test of {alpha_name} fails, error message: unsupported category {category} | "
                       f"Err-msg: {traceback.format_exc()}",
                       to=[telephone_dict[researcher]])
        logger.error(f"Test of {alpha_name} fails, error message: unsupported category {category}")
        dist.set_uuid(uuid, -1, time_out=time_out)
        continue

    # logger
    logger = getLogger("logstash-ft_system",
                       {
                           "task_id": result["jobid"],
                           "project_name": result["project_name"],
                           "branch_name": result["branch_name"],
                           "worker": socket.gethostname(),
                           "process": "ENROLLMENT TEST",
                       })

    df = dist.context.deserialize(result['feature_data']).dropna(subset=[alpha_name])
    FT.logger = logger
    alpha_in_db = FT.features_in_db

    if alpha_name in alpha_in_db:
        logger.error(f"Feature {alpha_name} already in feature db, no action...")
        ding.send_ding("ERROR | PLATFORM_UPLOAD_SERVICE",
                       f"Feature {alpha_name} already in feature db, no action...")
        dist.set_uuid(uuid, -4, time_out=time_out)
        continue

    else:
        try:
            logger.info("=" * 125)
            logger.info(f"Test of {alpha_name} starts...")
            status = FT.distribute_upload_feature_from_dataframe(researcher, df,
                                                                 universe='Investable',
                                                                 timedelta=None,
                                                                 output=output_folder, category=category,
                                                                 verbose=True)
            dist.set_uuid(uuid, result_info[status], time_out=time_out)

        except Exception as err:
            ding.send_ding("ERROR | PLATFORM_UPLOAD_SERVICE",
                           f"Test of {alpha_name} fails, error message: {err} | Err-msg: {traceback.format_exc()}",
                           to=[telephone_dict[researcher]])
            logger.error(f"Test of {alpha_name} fails, error message: {err}")
            dist.set_uuid(uuid, -1, time_out=time_out)
            continue
