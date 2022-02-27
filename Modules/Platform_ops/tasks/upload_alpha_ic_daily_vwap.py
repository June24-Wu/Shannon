import socket
from config.config import TEST_FACTORS_DAILY_TI0, FACTORS_PATH, FACTOR_START_DATE, TOP_MYSQL_TIO
from Research.utils.namespace import load_namespace
from Platform_ops.feature.ops import FeatureOps
from Research.feature.ft import FeatureAnalysis
import os
benchmark = 'Investable'
bt_price = 'vwap'
create_alpha_ic_daily_close = f"""
    CREATE TABLE if not exists alpha_ic_daily_{benchmark}_{bt_price} (
        alpha_name varchar(96) NOT NULL,
        trading_time datetime NOT NULL,
        IC_5min float NOT NULL,
        IC_15min float NOT NULL,
        IC_30min float NOT NULL,
        IC_60min float NOT NULL,
        IC_120min float NOT NULL,
        IC_1d float NOT NULL,
        IC_2d float NOT NULL,
        IC_3d float NOT NULL,
        IC_4d float NOT NULL,
        IC_5d float NOT NULL,
        IC_10d float NOT NULL,
        IC_20d float NOT NULL,
        create_time timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (`alpha_name`, `trading_time`)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4"""

start_date = '2016-07-01'
end_date = '2021-07-01'

config_path = r'../config/feature_bt_template_fangxin'
print('Loading the configuration from ' + config_path)
configs = load_namespace(config_path)
FT = FeatureOps(configs, TOP_MYSQL_TIO, logger=True,
                check_table=False, check_corr=False, feature_path=FACTORS_PATH)
ft = FeatureAnalysis(configs)

if FT.logger is not None:
    FT.logger.info("=" * 125)
    FT.logger.info(f"Task of feature test starts, worker: {socket.gethostname()}...")
conn = FT.db_api.pool.connection()
curs = conn.cursor()
curs.execute(create_alpha_ic_daily_close)

factors_info = ft.db_api.get_factor_info()
feature_names = factors_info[factors_info['researcher'] == 'fangxin']['table_name'].values.tolist()
for alpha_name in feature_names[610:]:
    print(alpha_name)
    FT.return_path = os.path.join(FT.options.return_path_daily, '5min')
    FT.load_feature_from_file(alpha_name,  start_date, end_date, universe='Investable')
    FT.generate_alpha_daily_ic_from_file(alpha_name, benchmark, bt_price, tp=6)
