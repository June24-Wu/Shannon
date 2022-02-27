import numpy as np
import pandas as pd
from Research.feature.ft import FeatureAnalysis

from utils import namespace as namespace
from utils import normalization as norm

pd.set_option('display.width', 500)
pd.set_option('display.max_columns', 30)
pd.set_option('display.max_rows', 300)


def func(sr):
    return np.where(abs(sr) >= sr.abs().quantile(0.9), sr.mean(), sr)


class Test:
    config_path = r'/home/liguichuan/Desktop/Project/release/research/config/feature_bt_template'

    @staticmethod
    def test_load_feature_data_form_file():
        feature_path = r"/home/ShareFolder/factor_lib"
        config_path = Test.config_path
        print('Loading the configuration from ' + config_path)
        configs = namespace.load_namespace(config_path)
        configs.universe = 'All'
        FT = FeatureAnalysis(configs, feature_path=feature_path)

        alpha_list = ['alpha_buy_pressure_filter']
        print(alpha_list)
        start_date = '2016-06-01'
        end_date = '2021-06-30'

        # FT.load_feature_from_file(alpha_list, start_date, end_date, universe='Investable',
        #                           timedelta=None, transformer=func)
        # df = FT.feature_data.reset_index().rename(columns={'ticker': 'symbol', "alphaONC": "alpha_onc_revised"})
        # output = '/home/ShareFolder/factor_lib/ti0/liguichuan'
        # convert_to_standard_daily_feature_csv("alpha_onc_revised", df, output)

        FT.load_feature_from_file(alpha_list, start_date, end_date, universe='Investable',
                                  timedelta=None, transformer=norm.standard_scale)
        FT.load_return_data(transformer=None, price='close')

        for alpha_name in alpha_list:
            FT.get_intersection_ic(feature_name=alpha_name, truncate_fold=None, method='spearman', period=('1d',))
            ic_flag, ic_value = FT.test_ic(alpha_name)
            df, df_all = FT.get_ic_summary_by_month(num=6)

            trading_direction = int(np.sign(ic_value))
            if trading_direction == -1:
                negative = True
            else:
                negative = False
            # FT.get_top_return(alpha_name, negative=negative, trade_type='long-only', transmission_rate=0.0000,
            #                   stock_pct=0.1, tax_rate=0.000, period=12, verbose=False)
            import matplotlib.pyplot as plt
            df, holdings = FT.get_group_returns(alpha_name, negative=negative, group_num=20, transmission_rate=0.000, tax_rate=0.001)
            # print(df)
            # print(holdings[9][20210629])
            y_data_1 = df.alpha_group0
            y_data_2 = df.alpha_group1
            y_data_3 = df.alpha_group2
            y_data_4 = df.alpha_group3
            y_data_5 = df.alpha_group4
            y_data_6 = df.alpha_group5
            y_data_7 = df.alpha_group6
            y_data_8 = df.alpha_group7
            y_data_9 = df.alpha_group8
            y_data_10 = df.alpha_group19
            fig4 = plt.figure(f'Figure', figsize=(20, 10)).add_subplot(111)
            x_data_slice = pd.to_datetime(y_data_1.index.astype(str))
            fig4.plot(x_data_slice, y_data_1, label='group_0')
            fig4.plot(x_data_slice, y_data_2, label='group_1')
            fig4.plot(x_data_slice, y_data_3, label='group_2')
            fig4.plot(x_data_slice, y_data_4, label='group_3')
            fig4.plot(x_data_slice, y_data_5, label='group_4')
            fig4.plot(x_data_slice, y_data_6, label='group_5')
            fig4.plot(x_data_slice, y_data_7, label='group_6')
            fig4.plot(x_data_slice, y_data_8, label='group_7')
            fig4.plot(x_data_slice, y_data_9, label='group_8')
            fig4.plot(x_data_slice, y_data_10, label='group_9')
            fig4.set_ylabel('return')
            fig4.legend()
            fig4.set_title(f'{alpha_name}')
            plt.show()

    @staticmethod
    def test_feature_corr():
        feature_path = r"/home/ShareFolder/factor_lib"
        config_path = self.config_path
        print('Loading the configuration from ' + config_path)
        configs = namespace.load_namespace(config_path)
        configs.universe = 'Investable'
        FT = FeatureAnalysis(configs, feature_path=feature_path)
        start_date = '2018-07-01'
        end_date = '2020-06-30'

        alpha_list = list(FT.features_in_path.keys())[0:2]
        FT.load_feature_from_file(alpha_list, start_date, end_date, universe='Investable',
                                  timedelta=None, transformer=norm.standard_scale)

        for alpha_name in alpha_list:
            corr_result = FT.get_correlation_with_factor_lib(feature_name=alpha_name, method='spearman')
            print(corr_result)

    @staticmethod
    def test_load_feature_data_form_dataframe():
        feature_path = r"/home/ShareFolder/factor_lib"
        config_path = self.config_path
        print('Loading the configuration from ' + config_path)
        configs = namespace.load_namespace(config_path)
        configs.universe = 'All'
        FT = FeatureAnalysis(configs, feature_path=feature_path)

        alpha_list = list(FT.features_in_path.keys())[0:2]
        start_date = '2016-07-01'
        end_date = '2018-06-30'

        _, cover_rate = FT.load_feature_from_file(alpha_list, start_date, end_date, universe='Investable',
                                                  timedelta=None, transformer=norm.standard_scale, cover_rate=True)
        print(FT.feature_data.sort_index().head())
        _, cover_rate = FT.load_feature_from_dataframe(FT.feature_data, universe='Investable', check_timestamp=False,
                                                       transformer=norm.standard_scale, cover_rate=True)
        print(FT.feature_data.sort_index().head())
        print(cover_rate)

    @staticmethod
    def test_base_lib():
        from database.mysql import BaseLib
        base_lib = BaseLib()
        df2 = base_lib.query_barra_factors('factor_exposure_hs300_hs300', '20170327', '20211109')
        print(df2)

    @staticmethod
    def test_top_return():
        config_path = self.config_path
        print('Loading the configuration from ' + config_path)
        configs = namespace.load_namespace(config_path)
        FT = FeatureAnalysis(configs, feature_path=r"/home/ShareFolder/factor_lib")
        alpha_list = list(FT.features_in_path.keys())[0:2]
        start_date = '2018-01-01'
        end_date = '2021-03-02'
        FT.load_feature_from_file(alpha_list, start_date, end_date, universe='Investable',
                                  timedelta=None, transformer=norm.standard_scale)
        FT.load_return_data()

        for alpha_name in alpha_list:
            FT.get_intersection_ic(feature_name=alpha_name, truncate_fold=None, method='spearman',
                                   period=('1d', '3d', '5d'))
            ic_flag, trading_direction = FT.test_ic(alpha_name, verbose=False)
            df, df_all = FT.get_ic_summary_by_month(num=6)
            if trading_direction == -1:
                negative = True
            else:
                negative = False
            FT.get_top_return(alpha_name, negative=negative, trade_type='long-short', transmission_rate=0.0,
                              tax_rate=0.0, verbose=True)

    @staticmethod
    def test_get_correlation_within_features():
        feature_path = r"/home/ShareFolder/factor_lib"
        config_path = self.config_path
        print('Loading the configuration from ' + config_path)
        configs = namespace.load_namespace(config_path)
        configs.universe = 'Investable'
        FT = FeatureAnalysis(configs, feature_path=feature_path)

        alpha_list = list(FT.features_in_path.keys())[0:5]
        start_date = '2016-01-01'
        end_date = '2021-07-31'

        FT.load_feature_from_file(alpha_list, start_date, end_date, universe='Investable',
                                  timedelta=None, transformer=norm.standard_scale)
        corr_table = FT.get_correlation_within_features(alpha_list[0], start_time='2020-08-01', end_time='2021-07-31')
        print(corr_table)

    @staticmethod
    def test_evaluate_feature_from_file():
        feature_path = r"/home/ShareFolder/factor_lib"
        config_path = self.config_path
        print('Loading the configuration from ' + config_path)
        configs = namespace.load_namespace(config_path)
        configs.universe = 'Investable'
        FT = FeatureAnalysis(configs, feature_path=feature_path)

        start_date = '2020-06-01'
        end_date = '2021-06-30'

        alpha_name = list(FT.features_in_path.keys())[0]
        stauts, change_reason, msg = FT.evaluate_feature_from_file(alpha_name, start_date, end_date,
                                                                   universe='Investable',
                                                                   transformer=norm.standard_scale,
                                                                   verbose=True, output=None)
        print(stauts, change_reason, msg)

    @staticmethod
    def test_evaluate_feature_from_dataframe():
        feature_path = r"/home/ShareFolder/factor_lib"
        config_path = self.config_path
        print('Loading the configuration from ' + config_path)
        configs = namespace.load_namespace(config_path)
        configs.universe = 'Investable'
        FT = FeatureAnalysis(configs, feature_path=feature_path)

        start_date = '2020-06-01'
        end_date = '2021-06-30'

        alpha_name = list(FT.features_in_path.keys())[0]
        FT.load_feature_from_file(alpha_name, start_date, end_date, universe='Investable',
                                  timedelta=None, transformer=norm.standard_scale)
        df = FT.feature_data
        print(df.head())
        stauts, change_reason, msg = FT.evaluate_feature_from_dataframe(alpha_name, df,
                                                                        universe='Investable',
                                                                        transformer=norm.standard_scale,
                                                                        verbose=True, output=None)
        print(stauts, change_reason, msg)


if __name__ == '__main__':
    t = Test()
    t.test_load_feature_data_form_file()
