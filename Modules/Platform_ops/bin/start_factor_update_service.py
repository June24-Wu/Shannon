import socket
import time
import traceback

from DevTools.tools.ding import Ding
from DevTools.tools.elk_logger import get_elk_logger
from Platform_ops.feature.ops import Distribute, FeatureOps

from config.config import TEST_FACTORS_DAILY_TI0, configs, telephone_dict

logger = get_elk_logger("platform_upload_service")
ding = Ding(configs.DING_SECRET_KEY, configs.DING_ACCESS_TOKEN)
FT = FeatureOps(configs, TEST_FACTORS_DAILY_TI0, logger=True,
                check_table=True, check_corr=False, feature_path=configs.FACTORS_PATH)
dist = Distribute(configs)


logger.info("=" * 125)
logger.info(f"Daily operation feature starts, worker: {socket.gethostname()}...")
ding.send_ding("INFO | PLATFORM_UPDATE_SERVICE",
               f"Task of feature upload starts, worker: {socket.gethostname()}...")

while True:
    result = dist.get_one_factor_from_redis_for_dataframe_update()
    if result is None:
        time.sleep(10)
        continue
    researcher = result['uname']
    df = dist.context.deserialize(result['feature_data'])
    uuid = result['uuid']
    alpha_name = result['alpha_name']

    try:
        logger.info("=" * 125)
        logger.info(f"Daily operation of {alpha_name} starts...")
        FT.distribute_update_feature_from_dataframe(researcher, df)
        dist.set_uuid(uuid, 1)

    except Exception as err:
        ding.send_ding("ERROR | PLATFORM_UPDATE_SERVICE",
                       f"Test of {alpha_name} fails, error message: {err} | Err-msg: {traceback.format_exc()}",
                       to=[telephone_dict[researcher]])
        print(traceback.format_exc())
        dist.set_uuid(uuid, -1)
        logger.error(f"Daily operation of {alpha_name} fails, error message: {err}")
        continue
