import datetime
import numpy as np
import pandas as pd
import torch
import DataAPI
import os
import Research.utils.namespace as namespace
from tqdm import tqdm
from Research.feature.ft import FeatureAnalysis

def convert_to_standard_daily_data_par(df: pd.DataFrame, output_name: str, output_path: str):
    grouped = df.groupby('timestamp')
    for date, group in tqdm(grouped):
        date_format = pd.to_datetime(date).date()
        assert DataAPI.is_trading_day(date), f"{date} is not a trading date!"
        file_name = datetime.date.strftime(date_format, '%Y%m%d') + '.par'
        folder = os.path.join(output_path, output_name, str(date_format.year))
        if not os.path.exists(folder):
            os.makedirs(folder)
        file = os.path.join(folder, file_name)
        group.to_parquet(file)
    return None


def convert_to_standard_daily_data_csv(df: pd.DataFrame, output_name: str, output_path: str):
    if len(list(df.index.names)) != 2:
        raise ValueError(r'Please set_index(["timestamp","ticker"]')
    grouped = df.groupby('timestamp')
    for date, group in tqdm(grouped):
        date_format = pd.to_datetime(date).date()
        assert DataAPI.is_trading_day(date), f"{date} is not a trading date!"
        file_name = datetime.date.strftime(date_format, '%Y%m%d') + '.csv'
        folder = os.path.join(output_path, output_name, str(date_format.year))
        if not os.path.exists(folder):
            os.makedirs(folder)
        file = os.path.join(folder, file_name)
        group.to_csv(file,header=False, encoding='utf-8')
    return None


def concat_original_data(alpha_name, alpha_list, start_date = "2016-01-01", end_date = "2021-06-01",
                         output_path="/home/wuwenjun/Data/", universe='Investable',corr_filter = False,base_list = []):
    if corr_filter == False:
        config_path = r'/home/ShareFolder/lgc/Modules/Research/config/feature_bt_template'
        configs = namespace.load_namespace(config_path)
        print(alpha_list)
        FT = FeatureAnalysis(configs, feature_path=r"/home/ShareFolder/feature_platform")
        feature_data = FT.load_feature_from_file(alpha_list, start_date, end_date, universe=universe, timedelta=None)[0]
    else:
        feature_data , alpha_list = __feature_filter(alpha_list,start_date,end_date,bench_mark=corr_filter,base_list=base_list)
    feature_data[feature_data.columns] = feature_data[feature_data.columns].astype("float32")
    feature_data.dropna(axis=0, inplace=True)
    convert_to_standard_daily_data_par(df=feature_data, output_name=alpha_name, output_path=output_path)
    return feature_data , alpha_list

def __feature_filter(alpha_list, start_date, end_date, bench_mark = 0.9,base_list = []):
    config_path = r'/home/ShareFolder/lgc/Modules/Research/config/feature_bt_template'
    configs = namespace.load_namespace(config_path)
    FT = FeatureAnalysis(configs, feature_path=r"/home/ShareFolder/feature_platform")
    FT.load_feature_from_file(alpha_list+base_list, start_date, end_date, universe='Investable',
                              timedelta=None)
    for i in alpha_list:
        if base_list == []:
            base_list.append(i)
            continue
        corr_table = FT.get_correlation_within_features(i, start_time=start_date, end_time=end_date, others=base_list)
        if corr_table["correlation"].max() <= abs(bench_mark * 100):
            print("add [%s]" % i)
            base_list.append(i)
        else:
            print("delete [%s] because of %d [%s]" % (i,corr_table["correlation"].max(),corr_table["correlation"].idxmax()))
    FT.feature_data = FT.feature_data[base_list]
    return FT.feature_data , base_list

def generate_alpha_list(feat_list, method, day):
    name_list = []
    # Moving Cov
    if "COV" in method:
        for i in range(len(feat_list) - 1):
            for j in range(i + 1, len(feat_list)):
                name_list.append("COV_%s_%s_%s" % (feat_list[i], feat_list[j], day))
    if "CORR" in method:
        for i in range(len(feat_list) - 1):
            for j in range(i + 1, len(feat_list)):
                name_list.append("CORR_%s_%s_%s" % (feat_list[i], feat_list[j], day))
    if "STD" in method:
        for i in range(len(feat_list)):
            name_list.append("STD_%s_%s" % (feat_list[i], day))
    if "ZSCORE" in method:
        for i in range(len(feat_list)):
            name_list.append("ZSCORE_%s_%s" % (feat_list[i], day))
    if "RETURN" in method:
        for i in range(len(feat_list)):
            name_list.append("RETURN_%s_%s" % (feat_list[i], day))
    if "DECAY" in method:
        for i in range(len(feat_list)):
            name_list.append("DECAY_%s_%s" % (feat_list[i], day))
    if "Mavg" in method:
        for i in range(len(feat_list)):
            name_list.append("Mavg_%s_%s" % (feat_list[i], day))
    return name_list

def generate_shift_data(alpha_name, shift,sequence, target, data_path="/home/wuwenjun/Data/"):
    dataloader = DataLoader()
    dataloader.load_data_from_file(alpha_name=alpha_name, data_path=data_path, end_date="2022-01-01")

    # generate shift list
    sequence_list = [0]
    shift_value = shift
    while len(sequence_list) < sequence:
        sequence_list.append(shift_value)
        shift_value += shift
    sequence_list.sort(reverse=True)
    print(sequence_list)
    # shift
    final_df = []
    for index, group in tqdm(dataloader.feature_data.groupby("ticker")):
        group_df = []
        for i in sequence_list:
            a = group.shift(i)
            a.columns = [str(j) + "_Shift_%i" % i for j in group.columns]
            group_df.append(a)
        group_df = pd.concat(group_df, axis=1)
        final_df.append(group_df)
    final_df = pd.concat(final_df, axis=0)

    # target
    configs = namespace.load_namespace(r'/home/ShareFolder/lgc/Modules/Research/config/feature_bt_template')
    FT = FeatureAnalysis(configs, feature_path=r"/home/ShareFolder/feature_platform")
    FT.load_feature_from_file(target, "2015-01-01", "2022-01-01", universe='Investable', timedelta=None)
    FT.feature_data.rename(columns={target: "target"}, inplace=True)

    final_df = pd.concat([final_df, FT.feature_data], axis=1)
    final_df.dropna(axis=0, inplace=True)
    output_name = alpha_name + "_Shift_%i_Sequence_%i_%s" % (shift,sequence,target)
    convert_to_standard_daily_data_par(df=final_df, output_name=output_name,
                                       output_path=data_path)
    return final_df


class DataLoader(object):
    def __init__(self):
        self.feature_data = None
        self.target = None
        self.length = None
        self.feature_length = None
        self.sequence = None
        self.y = None
        self.shape = None

    def load_data_from_file(self,alpha_name, end_date,data_path = "/home/wuwenjun/Data_lib/ti0/wuwenjun/", start_date="2015-01-01"):
        time_list = DataAPI.get_trading_days(start_date, end_date)
        final = []
        for date in tqdm(time_list):
            date = data_path + alpha_name + "/" + str(date.year) + '/' + str(date.year) + str(date.month).zfill(2) + str(date.day).zfill(
                2) + ".par"
            if (os.path.exists(date)):
                final.append(pd.read_parquet(date))
        self.feature_data = pd.concat(final, axis=0)
        self.length = self.feature_data.shape[0]
        return self.feature_data

    def to_torch_DataLoader(self,sequence, shuffle, batch_size=1024, num_workers=16):
        self.target = pd.DataFrame(self.feature_data["target"])
        loader = self.feature_data.drop("target", axis=1)
        self.sequence = sequence
        # self.feture_data.set_index(["timestamp", "ticker"], inplace=True)
        x = torch.from_numpy(np.array(loader).reshape(self.length, sequence,-1 ))
        self.shape = x.shape
        self.feature_length = x.shape[2]
        y = torch.from_numpy(np.array(self.target).reshape(-1, 1))
        loader = torch.utils.data.TensorDataset(x, y)
        loader = torch.utils.data.DataLoader(
            dataset=loader,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True
        )
        return loader


