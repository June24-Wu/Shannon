from Platform_ops.feature.ops import Distribute
from Research.utils.namespace import load_namespace
from Research.feature.ft import FeatureAnalysis
import Research.utils.normalization as norm
config_path = r'/config/feature_bt_template_test'
configs = load_namespace(config_path)
dist = Distribute(configs)

feature_path = r"/home/ShareFolder/factor_lib"
config_path = r'/home/liguichuan/Desktop/Project/release/platform/research/config/feature_bt_template'
print('Loading the configuration from ' + config_path)
configs = load_namespace(config_path)
FT = FeatureAnalysis(configs, feature_path=feature_path)

alpha_list = list(FT.features_in_path.keys())[0]
start_date = '2017-01-01'
end_date = '2021-07-01'

FT.load_feature_from_file(alpha_list, start_date, end_date, universe='Investable',
                          timedelta=None, transformer=norm.standard_scale)
df = FT.feature_data
dist.add_factors_to_redis_for_dataframe(researcher='wuwenjun', dataframe_list=[df], clear_queue=True, category='PV',
                                        file_path='abs')
# dist.distribute_correlation_calculator()
