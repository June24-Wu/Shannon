import time
from Platform.backtest.bt import BTDaily
import Platform.utils.namespace as namespace


def test():
    config_path = r'/home/ShareFolder/lgc/Modules/Platform/backtest/config/template'
    print('Loading the configuration from ' + config_path)
    configs = namespace.load_namespace(config_path)
    configs.trading_type = 'long-only'
    configs.data_format = 'par'
    configs.ti = 36
    configs.trade_period = 12
    configs.stock_percentage = False
    configs.stock_num = 200
    configs.transmission_rate = 0.0005
    configs.benchmark = "ZZ500"
    configs.bt_price = 'vwap'
    configs.universe = "Investable"
    configs.score_path = r'/home/ShareFolder/lgc/Modules/Platform/backtest/demo/test_data/par'

    bt = BTDaily(configs, start_date='2017-01-01', end_date='2020-11-30')
    start = time.time()
    bt.run()
    df, df_final = bt.evaluate(evalRange=(
        (20170101, 20171231), (20180101, 20181231), (20190101, 20191231), (20200101, 20201130)))

    end = time.time()
    print('Time consume: %.2f seconds\n' % (end - start))


if __name__ == '__main__':
    test()
