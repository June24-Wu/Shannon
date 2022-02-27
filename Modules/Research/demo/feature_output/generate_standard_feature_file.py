import time
from Platform.database.mysql import MysqlAPI
from Platform.utils.persistence import convert_to_standard_daily_feature_csv, convert_to_standard_daily_feature_par
from Platform.config.mysql_info import FACTOR_LIB_MYSQL_TIO


def test():
    alpha_name = "pv_daily_xu_gtja__alpha60"
    start_time = time.time()
    print("Fetch data from database...")
    mysql_api = MysqlAPI(FACTOR_LIB_MYSQL_TIO)
    df = mysql_api.query(alpha_name, '2020-06-01', '2020-06-30 23:59:59')
    # df = mysql_api.query_factors(alpha_name, '2020-06-01', '2020-06-30 23:59:59')
    end_time = time.time()
    print('DataFrame shape: %d | Time consume: %.2f seconds\n' % (df.shape[0], end_time - start_time))

    # correct generate time
    df.reset_index(inplace=True)
    df.rename(columns={'trading_time': 'timestamp', 'alpha_value': alpha_name}, inplace=True)
    df = df.reindex(columns=['symbol', 'timestamp', alpha_name])

    # output
    output_path = r'/home/ShareFolder/lgc/Modules/Platform/data_demo'
    convert_to_standard_daily_feature_csv(alpha_name, df, output_path)


if __name__ == '__main__':
    test()
