import multiprocessing as mp
import time
import pandas as pd
from tqdm import tqdm
import datetime
import warnings
warnings.filterwarnings('ignore')
from datetime import timedelta

def test(ticker):
    one_data = df[df['ticker'] == ticker]
    one_data.set_index(['timestamp','ticker'],inplace=True)
    target_df = one_data[return_value]
    one_data.drop(return_value,axis=1,inplace=True)
    concat_df = pd.DataFrame()
    for day in range(29,-1,-1):
        concat_df = pd.concat([concat_df,one_data.shift(day)],axis=1)
    concat_df.dropna(axis=0,how="any",inplace=True)
    concat_df = pd.concat([concat_df,target_df],axis=1,join='inner')
#     concat_df.to_csv(output_path + str(ticker) +".csv")
    return concat_df

def save(start_time,end_time):
    print("Start Save to" + 
        output_path + datetime.datetime.strptime(str(start_time), '%Y-%m-%d %H:%M:%S').strftime('%Y-%m-%d') + "_" +
                      datetime.datetime.strptime(str(end_time), '%Y-%m-%d %H:%M:%S').strftime('%Y-%m-%d') + ".par")
    final_df[
        (final_df['timestamp'] >= start_time) &
        (final_df['timestamp'] < end_time)].to_parquet(
        output_path + datetime.datetime.strptime(str(start_time), '%Y-%m-%d %H:%M:%S').strftime('%Y-%m-%d') + "_" +
                      datetime.datetime.strptime(str(end_time), '%Y-%m-%d %H:%M:%S').strftime('%Y-%m-%d') + ".par"
    )
    return None

if __name__=="__main__":
    return_value = '5d_ret'
    time_list = [20140101,20180101, 20180630, 20181231, 20190630, 20191231, 20200601, 20201231, 20210630]
    df = pd.read_csv("/home/wuwenjun/Data/AlphaNet_Original_Input.csv")
    output_path = '/home/wuwenjun/Data/AlphaNet_Original_Input/'

    num_cores = int(mp.cpu_count())
    pool = mp.Pool(num_cores)
    result = {}
    print("Start multiprocessing")
    t1 = time.time()
    for ticker in tqdm(df['ticker'].drop_duplicates()):
        result[ticker] = (pool.apply_async(test, args=(ticker,)))   #维持执行的进程总数为10，当一个进程执行完后启动一个新进程.
    pool.close()
    pool.join()
    t2 = time.time()
    print("running time", int(t2 - t1))
    df_list = []
    for i in tqdm(result):
        df_list.append(result[i].get())
    final_df = pd.concat(df_list)

    # preprocessing
    final_df.reset_index(inplace=True)
    final_df['timestamp'] = pd.to_datetime(final_df['timestamp'])
    final_df.columns = [str(i) for i in range(final_df.shape[1])]
    final_df.rename(columns={'0': 'timestamp', '1': 'ticker', str(final_df.shape[1] - 1): 'target'}, inplace=True)

    # Save Data
    print("Start Save Data")
    num_cores = int(mp.cpu_count())
    pool = mp.Pool(num_cores)
    result = {}
    t1 = time.time()
    for i in range(len(time_list) - 1):
        start_time = pd.to_datetime(str(time_list[i]))
        end_time = pd.to_datetime(str(time_list[i + 1]))
#         save(start_time,end_time)
        pool.apply_async(save, args=(start_time,end_time))
    pool.close()
    pool.join()
    t2 = time.time()
    print("running time", int(t2 - t1))