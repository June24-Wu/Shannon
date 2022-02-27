import sys, os

import pandas as pd

import numpy as np
import DataAPI as api
from functools import wraps

from DataAPI import get_trading_days, get_next_trading_day, get_universe, get_stock_daily_return, convert_to_date

from Platform.database.mysql import BaseLib

import multiprocessing as mp



base_lib = BaseLib(logger=None)





def fwd_reader(fwd=True):

    if fwd:

        def read_fwd(func):

            @wraps(func)

            def t(start, end, *args, **kwargs):

                start2 = get_trading_days(start, count=2)[-1]

                end2 = get_next_trading_day(end)

                df = func(start2, end2, *args, **kwargs)

                df2 = df.index + pd.Timedelta(days=-1)

                df2 = df2.stack()

                return df2.rename_axis([None, None])



            return t



        return read_fwd

    else:

        def read_normal(func):

            @wraps(func)

            def t(*args, **kwargs):

                df = func(*args, **kwargs)

                return df.rename_axis([None, None])



            return t



        return read_normal





def read_data(path, start, end, header=None, names=None, index_col='ticker', sep=',', dtype=None, **kwargs):

    files = [(date, os.path.join(path, str(date.year), str(date).replace('-', '') + '.csv'))

             for date in get_trading_days(start, end)]



    df_list = []

    for date, file in files:

        df_temp = pd.read_csv(file, header=header, names=names, sep=sep, dtype=dtype, **kwargs)

        df_temp.insert(0, 'date', date)

        df_temp.set_index(['date', index_col], inplace=True)

        df_list.append(df_temp)

    df = pd.concat(df_list)

    return df





@fwd_reader(False)

def get_portfolio_weight(start, end, file, file_type='csv', sep=',', header=None,

                         names=['ticker', 'weight'],

                         index_col='ticker', dtype={'ticker': str},

                         **kwargs):

    """

    path: csv dir

    start,end: yyyymmdd

    file_type: None -> no suffix

               'csv' -> .csv

               etc

    sep: seperator

    """

    assert file_type in ('series', 'csv'), f"Unsupported file_type: {file_type}!"



    if file_type == 'series':

        assert isinstance(file, pd.Series), f"Expect w to be a Series, but a type {type(file)} received!"

        w = file.rename_axis([None, None])

        w = w.loc[(w.index.get_level_values(0) >= int(convert_to_date(start).strftime('%Y%m%d'))) &

                  (w.index.get_level_values(0) <= int(convert_to_date(end).strftime('%Y%m%d')))]

        return w



    else:

        w = read_data(

            path=file

            , file_type=file_type

            , start=start, end=end

            , header=header

            , names=names, index_col=index_col

            , sep=sep

            , dtype=dtype

            , **kwargs

        )

        return w['weight']





@fwd_reader(False)

def get_index_weight(start, end, index='ZZ500'):

    """

    index in 'ZZ500' 'SZ50' 'HS300'

    """

    days = get_trading_days(start, end)

    df_list = [get_universe(date, universe=index)['weight'] for date in days]

    df = pd.concat(df_list)

    return df





@fwd_reader(False)

def get_factor_ret(universe, start, end):

    universe_mapper = {'hs300': 'factor_returns_hs300', 'zz500': 'factor_returns_zz500',

                       'investable': 'factor_returns_investable'}

    table_name = universe_mapper[universe]

    df = base_lib.query_barra_returns(table_name=table_name, start_time=start, end_time=end).drop('create_date',

                                                                                                  axis=1)

    df.index.name = 'date'

    return df.stack()





@fwd_reader(False)

def get_factor_exposure(universe, start, end):

    universe_mapper = {'hs300': 'factor_exposure_hs300_hs300', 'zz500': 'factor_exposure_zz500_zz500',

                       'investable': 'factor_exposure_investable_zz800'}

    table_name = universe_mapper[universe]

    df = base_lib.query_barra_factors(table_name=table_name, start_time=start, end_time=end).reset_index().drop(

        'create_date',

        axis=1).rename(columns={'dt_int': 'date'})

    return df.set_index(['date', 'symbol'])





@fwd_reader(False)

def get_ret(start, end):

    # r = get_feature('origin',start,end,names=['close_re'])['close_re']

    days = get_trading_days(start, end)

    # df_list = [get_stock_daily_return(date, ti=0, period='1d').reset_index()
    #
    #            for date in days]
    #
    # df = pd.concat(df_list)
    #
    # df['date'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%d').dt.strftime('%Y%m%d').astype(int)

    # df = df.set_index(['date', 'ticker'])

    # return df['1d_ret']

    df = api.get_stock_prices(ticker = None, start_date = start, end_date = end, fields = 'pct_chg',fq=True)

    return df['pct_chg']





def calc_portfolio_return(weight, fwdret, top=None):

    days = fwdret.index.remove_unused_levels().levels[0]

    res = []

    for dt in days:

        subw = weight.loc[dt]

        subw = subw.loc[subw.gt(0)]

        if top is not None:

            subw = subw.nlargest(top)

        subret = fwdret.loc[dt]

        valid_stock = subw.index | subret.index

        assert len(valid_stock) > 0, 'no valid stock when calculating portfolio return'

        subw = subw.reindex(index=valid_stock).fillna(0)

        subw /= subw.abs().sum()

        subret = subret.reindex(index=valid_stock).fillna(0)

        exreturn = subret.mul(subw).sum()

        res.append(exreturn)

    return pd.Series(res, index=days)





def calc_portfolio_fac_expo(weight, fac_expo, top=None):

    expo = []

    days = weight.index.remove_unused_levels().levels[0]

    for dt in days:

        subfac = fac_expo.loc[dt]

        subw = weight.loc[dt]

        subw = subw.loc[subw.gt(0)]

        if top is not None:

            subw = subw.nlargest(top)

        valid_stock = subw.index | subfac.index

        valid_factor = subfac.columns[~subfac.isna().all(axis=0)]

        assert (len(valid_stock) > 0) and (

                len(valid_factor) > 0), 'no valid stock when calculating portfolio factor return'

        subfac = subfac.reindex(index=valid_stock, columns=valid_factor).fillna(0)

        subw = subw.reindex(index=valid_stock).fillna(0)

        subw /= subw.abs().sum()

        factor_exposure = subfac.mul(subw, axis=0).sum()

        expo.append(factor_exposure.rename(dt))

    exposure = pd.concat(expo, axis=1)

    return exposure.unstack().rename_axis([None, None])





def calc_portfolio_turnover(weight, top=None):

    days = weight.index.remove_unused_levels().levels[0]

    change_rate = pd.Series()

    for i, dt in enumerate(days):

        subw = weight.loc[dt]

        subw = subw.loc[subw.gt(0)]

        if top is not None:

            subw = subw.nlargest(top)

        if i == 0:

            lstw = subw

        else:

            cinx = subw.index | lstw.index

            subw_nl = subw.reindex(cinx).fillna(0) / subw.sum()

            lstw_nl = lstw.reindex(cinx).fillna(0) / lstw.sum()

            change_rate.loc[dt] = subw_nl.sub(lstw_nl).abs().sum()

            lstw = subw

    return change_rate





mmm = lambda df: df.unstack().agg([np.mean, max, min, sum]).T





def mmm2(df):

    df2 = mmm(df)

    inx = np.argmax([df2['min'].abs(), df2['max'].abs()], axis=0)

    sgn = pd.concat([np.sign(df2['min'].loc[inx == 0]), np.sign(df2['max'].loc[inx == 1])]).reindex(df2.index)

    value = np.max([df2['min'].abs(), df2['max'].abs()], axis=0)

    df2['diffmax'] = sgn * value

    return df2





def make_table1(fwdfacret, delta_expo, delta_fret, output1=None):

    table = pd.concat((mmm2(x) for x in (fwdfacret, delta_expo, delta_fret)), keys=('FacRet', 'FacExExpo', 'FacExRet'),

                      axis=1).dropna(how='any')

    if output1 is not None:

        table.to_csv(output1, float_format='.%3f')

    critical = [('FacRet', 'sum'), ('FacExExpo', 'mean'), ('FacExExpo', 'diffmax'), ('FacExRet', 'sum')]

    subtable = table.loc[:, critical].sort_values(by=('FacExRet', 'sum'), ascending=False)

    print('\n========================================================================\n')

    print(

        'Factor Attribution\n\n`FacRet`: factor return \n`FacExExpo`: excess factor exposure \n`FacExRet`: excesse return gained by exposure\n')

    try:

        from tabulate import tabulate

        h = ['Factors \\ Keys'] + list(map('\n'.join, subtable.columns.tolist()))

        print(tabulate(subtable, headers=h, tablefmt='grid', floatfmt=".3f"))

    except:

        pd.options.display.float_format = '{:,.3f}'.format

        print(subtable.to_string())

        print('\nInstall `tabulate` for better print format')

    return table





def make_table2(delta_ret, fexplained, uexplained, output2=None):

    print('\n========================================================================\n')

    print(

        'Excess Return Decomposition\n\n`ExRet`: excess return \n`Explained`: excess return explained by factors\n`Unexplained`: excesse return unexplained\n')

    df = pd.concat([delta_ret, fexplained, uexplained], axis=0, keys=['ExRet', 'Explained', 'Unexplained']).swaplevel()

    table2 = mmm(df)

    if output2 is not None:

        table2.to_csv(output2, float_format='%.3f')

    try:

        from tabulate import tabulate

        h = ['Parts \\ Stats'] + table2.columns.tolist()

        print(tabulate(table2, headers=h, tablefmt='grid', floatfmt=".3f"))

    except:

        pd.options.display.float_format = '{:,.3f}'.format

        print(table2.to_string())

        print('Install `tabulate` for better formatting')

    return table2





def make_table3(turnover, cost_rate, output):

    print('\n========================================================================\n')

    print(f'Turnover & Cost\n\nturnover cost rate {cost_rate}\n')

    cost = turnover * cost_rate

    df = pd.concat([turnover, cost], axis=0, keys=['Turnover', 'Cost']).swaplevel()

    table2 = mmm(df)

    if output is not None:

        table2.to_csv(output, float_format='%.3f')

    try:

        from tabulate import tabulate

        h = ['Parts \\ Stats'] + table2.columns.tolist()

        print(tabulate(table2, headers=h, tablefmt='grid', floatfmt=".3f"))

    except:

        pd.options.display.float_format = '{:,.3f}'.format

        print(table2.to_string())

        print('Install `tabulate` for better formatting')





def wrapup(start, end, csv_path, index='zz500', top=None, output1=None, output2=None, output3=None, cost_rate=0.007,

           **kwargs):

    print('reading factor exposure ..')

    facexpo = get_factor_exposure(start, end)

    print('reading forward return ..')

    fwdret = get_ret(start, end)

    pweight = get_portfolio_weight(start, end, csv_path, **kwargs)

    iweight = get_index_weight(start, end, index=index)

    fwdfacret = get_factor_ret(start, end)



    pexpo = calc_portfolio_fac_expo(pweight, facexpo, top=top)

    iexpo = calc_portfolio_fac_expo(iweight, facexpo, top=None)

    pret = calc_portfolio_return(pweight, fwdret, top=top)

    iret = calc_portfolio_return(iweight, fwdret, top=None)

    turnover = calc_portfolio_turnover(pweight, top=top)



    delta_expo = pexpo - iexpo

    delta_fret = fwdfacret.mul(delta_expo).dropna()



    delta_ret = pret - iret

    fexplained = delta_fret.unstack().sum(axis=1)

    uexplained = delta_ret - fexplained



    print('\n========================================================================\n')

    days = get_trading_days(start, end)

    print(f'Data from {start} to {end} total {len(days)} trading days')

    print(f'compared with index {index}')



    make_table1(fwdfacret, delta_expo, delta_fret, output1)

    make_table2(delta_ret, fexplained, uexplained, output2)

    make_table3(turnover, cost_rate, output3)

    print('\n========================================================================\n')





class BarraTools:

    def __init__(self):

        self.pweight = None

        self.iweight = None

        self.facexpo = None

        self.fwdret = None

        self.fwdfacret = None

        self.delta_expo = None

        self.delta_fret = None

        self.delta_ret = None

        self.fexplained = None

        self.uexplained = None

        self.turnover = None

        self.Dates = None


    def load_data(self, start, end, file, universe, index, **kwargs):

        self.pweight = get_portfolio_weight(start, end, file, **kwargs)

        self.Dates = self.pweight.reset_index(level = 1).index.unique()

        self.iweight = get_index_weight(start, end, index=index)

        self.facexpo = get_factor_exposure(universe, start, end)

        self.fwdret = get_ret(start, end)
        self.fwdret = self.fwdret.loc[self.Dates]

        self.fwdfacret = get_factor_ret(universe, start, end)

        print('\n========================================================================\n')

        days = get_trading_days(start, end)

        print(f'Data from {start} to {end} total {len(days)} trading days')

        print(f'compared with index {index}')



    def calc_top(self, top=None):

        for df in (self.pweight, self.iweight, self.fwdret, self.facexpo, self.fwdfacret):

            assert df is not None, 'load_data first'



        pexpo = calc_portfolio_fac_expo(self.pweight, self.facexpo, top=top)

        iexpo = calc_portfolio_fac_expo(self.iweight, self.facexpo, top=None)

        pret = calc_portfolio_return(self.pweight, self.fwdret, top=top)

        iret = calc_portfolio_return(self.iweight, self.fwdret, top=None)

        self.turnover = calc_portfolio_turnover(self.pweight, top=top)

        self.delta_expo = pexpo - iexpo

        self.delta_fret = self.fwdfacret.mul(self.delta_expo).dropna().drop_duplicates()

        self.delta_ret = pret - iret

        self.fexplained = self.delta_fret.unstack().sum(axis=1)

        self.uexplained = self.delta_ret - self.fexplained



    def make_table_attr(self, output=None):

        for df in (self.fwdfacret, self.delta_expo, self.delta_fret):

            assert df is not None, 'load_data first then calc_top'



        self.table1 = make_table1(self.fwdfacret, self.delta_expo, self.delta_fret, output)



    def make_table_deco(self, output=None):

        for df in (self.delta_ret, self.fexplained, self.uexplained):

            assert df is not None, 'load_data first then calc_top'

        self.table2 = make_table2(self.delta_ret, self.fexplained, self.uexplained, output)



    def make_table_turn(self, cost_rate=0.007, output=None):

        for df in (self.turnover,):

            assert df is not None, 'load_data first then calc_top'

        make_table3(self.turnover, cost_rate, output)



    @property

    def Unexplained(self):

        return self.uexplained



    @property

    def Explained(self):

        return self.fexplained



    @property

    def ExRet(self):

        return self.delta_ret



    @property

    def FacExExpo(self):

        return self.delta_expo



    @property

    def FacExRet(self):

        return self.delta_fret



    @property

    def FacRet(self):

        return self.fwdfacret



    @property

    def Turnover(self):

        return self.turnover





def cal_barra_attribution(start_date, end_date, universe, file, file_type='series',

                          index='ZZ500', Top=None, output=None):

    table1, table2 = pd.DataFrame(), pd.DataFrame()

    result_dict = dict()

    if file_type == 'series':

        for i in file.columns:

            B = BarraTools()

            B.load_data(start=start_date, end=end_date, universe=str.lower(universe), file=file[i], file_type=file_type,

                        index=str.upper(index))

            B.calc_top(Top)



            if output is not None:

                output1 = str(output) + f'Portfolio_{i}_factor_attribution.csv'

                output2 = str(output) + f'Portfolio_{i}_excess_return_decomposition.csv'

            else:

                output1, output2 = None, None

            print('\n========================================================================\n')

            print('Portfolio :' + i)

            B.make_table_attr(output1)

            B.make_table_deco(output2)

            result_dict[i] = (B.table1.copy(), B.table2.copy())

        return result_dict

    else:

        B = BarraTools()

        B.load_data(start=start_date, end=end_date, universe=str.lower(universe), file=file,

                    file_type=file_type, index=str.upper(index))

        B.calc_top(Top)

        if output is not None:

            output1 = str(output) + f'Portfolio_{i}_factor_attribution.csv'

            output2 = str(output) + f'Portfolio_{i}_excess_return_decomposition.csv'

        else:

            output1, output2 = None, None

        B.make_table_attr(output1)

        B.make_table_deco(output2)





def test():

    path = r"/home/ShareFolder/wyf/BarraTools/sample.pickle"

    file = pd.read_pickle(path)

    B = BarraTools()

    B.load_data(start=20170103, end=20170428, universe='investable',

                file=file['1'], file_type='series', index='ZZ500')

    B.calc_top(top=200)

    B.make_table_attr()

    B.make_table_deco()

    # B.make_table_turn(output=None)





def test_1():

    import time

    time_start = time.time()

    path = r"/home/ShareFolder/wyf/BarraTools/sample.pickle"

    file = pd.read_pickle(path)

    cal_barra_attribution(20170103, 20170428, universe='ZZ500', file=file,

                          file_type='series', index='ZZ500', output=None)

    time_end = time.time()

    print('Time consume: %.2f' % (time_end - time_start))


def test_2():

    import time

    time_start = time.time()

    path = r"/home/ShareFolder/syt/DataSample/Pos_For_Barra.pickle"

    file = pd.read_pickle(path)

    x = cal_barra_attribution(20190106, 20191228, universe='hs300', file=file,

                          file_type='series', index='hs300', output=None)

    time_end = time.time()

    print('Time consume: %.2f' % (time_end - time_start))
	
    print(x['Wgt'])
    print(x['2'])

    




if __name__ == '__main__':

    test_2()

    # import argparse

    # parse = argparse.ArgumentParser()

    # parse.add_argument('start', help='start of calculating period', type=int)

    # parse.add_argument('end', help='end of calculating period', type=int)

    # parse.add_argument('csv_path', help='alpha csv path', type=str)

    # parse.add_argument('-t', '--top', help='top alpha value to filter code', type=int, default=None)

    # parse.add_argument('-c', '--cost', help='cost rate of turnover', type=float, default=0.007)

    # parse.add_argument('-i', '--indexname', help='index to compare', type=str, default='zz500',

    #                    choices=['sz50', 'zz500', 'hs300'])

    # parse.add_argument('-o1', '--output1', help='Factor Attribution result output path', type=str, default=None)

    # parse.add_argument('-o2', '--output2', help='Excess Return Decomposition result output path', type=str,

    #                    default=None)

    # parse.add_argument('-o3', '--output3', help='Turnover result output path', type=str, default=None)

    # args = parse.parse_args()

    # wrapup(start=args.start

    #        , end=args.end

    #        , csv_path=args.csv_path

    #        , index=args.indexname

    #        , top=args.top

    #        , cost_rate=args.cost

    #        , output1=args.output1

    #        , output2=args.output2

    #        , output3=args.output3)

