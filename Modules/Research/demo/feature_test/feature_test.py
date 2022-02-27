import Platform.utils.namespace as namespace
import Platform.utils.normalization as norm
import numpy as np

from feature.ft import FeatureAnalysis


def test():
    # config_path = r'/home/ShareFolder/lgc/Modules/Platform/config/feature_bt_template'
    config_path = r'/home/liguichuan/Desktop/Project/release/Platform/config/feature_bt_template'
    print('Loading the configuration from ' + config_path)
    configs = namespace.load_namespace(config_path)
    configs.universe = 'Investable'
    FT = FeatureAnalysis(configs, feature_path=r"/home/ShareFolder/lgc/data/test/factors/fangxin")

    alpha_list = FT.features_in_path[0:50]
    start_date = '2016-07-01'
    end_date = '2018-06-30'

    for alpha_name in alpha_list:
        FT.load_feature_from_file(alpha_list, start_date, end_date, universe='Investable',
                                  timedelta=None, transformer=norm.rank)
        print(FT.feature_data)
        quit()

        FT.load_return_data()
        FT.get_intersection_ic(feature_name=alpha_name, truncate_fold=None, method='spearman',
                               period=('1d', '3d', '5d'))
        ic_flag, ic_value = FeatureAnalysis.get_ic_test_result(FT.feature_names, FT.ic_table,
                                                               logger=FT.logger)
        df, df_all = FT.get_ic_summary_by_month(num=6)

        trading_direction = int(np.sign(ic_value))
        if trading_direction == -1:
            negative = True
        else:
            negative = False
        FT.get_top_return(negative=negative, trade_type='long-only', transmission_rate=0.00025, stock_pct=1,
                          tax_rate=0.001, period=12, verbose=True)

        # result, holdings = FT.get_group_returns(negative=negative)
        # print(result)
        # print(holdings[0])


if __name__ == '__main__':
    test()
