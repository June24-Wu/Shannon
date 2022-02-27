from feature.ops import Distribute
from Research.utils.namespace import load_namespace
from config.config import TOP_MYSQL_TIO
config_path = r'/config/feature_bt_template_test'
configs = load_namespace(config_path)
dist = Distribute(configs)
dist.add_all_factors_to_redis_for_file(mysql_info=TOP_MYSQL_TIO, check_factors_in_lib=True, researchers=['fangxin'])
dist.distribute_correlation_calculator()
