import pandas as pd
from os import walk
from tqdm import tqdm
data_path = "/home/wuwenjun/Data/AlphaNet_Original_Input/"
dataframe_list = pd.DataFrame()
for f, _, i in walk(data_path):
    for j in tqdm(i):
        print(f+j)
#         dataframe_list = pd.concat([dataframe_list, pd.read_parquet(f + "/" + j)], axis=0)
# dataframe_list['timestamp'] = pd.to_datetime(dataframe_list['timestamp'])