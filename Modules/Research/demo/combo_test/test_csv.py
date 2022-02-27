import time
from Platform.backtest.bt import BTDaily
import Platform.utils.namespace as namespace


def test():
    config_path = r'/home/ShareFolder/lgc/Modules/Platform/config/combo_bt_template'
    print('Loading the configuration from ' + config_path)
    configs = namespace.load_namespace(config_path)
    configs.trading_type = 'long-only'
    configs.ti = 0
    configs.trade_period = 0
    configs.stock_percentage = True
    configs.stock_num = 1.0
    configs.transmission_rate = 0.0005
    configs.benchmark = "ZZ500"
    configs.bt_price = 'close'
    configs.universe = "Investable"
    configs.score_path = r'/home/ShareFolder/syt/Data/score'
    # configs.keep_pos_percentile = 0.0

    bt = BTDaily(configs, start_date='2017-01-01', end_date='2020-12-31')
    start = time.time()
    bt.run()
    df, df_final = bt.evaluate(evalRange=(
        (20170101, 20171231), (20180101, 20181231), (20190101, 20191231), (20200101, 20201231)))

    end = time.time()
    print('Time consume: %.2f seconds\n' % (end - start))
    # print('Stock list of date {}:'.format(str(20200630)))
    # print(bt.holdings[20200630].index.tolist())


if __name__ == '__main__':
    test()
