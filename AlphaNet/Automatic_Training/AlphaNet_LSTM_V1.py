import time
import datetime
import os
from Research.backtest.bt import BTDaily
import matplotlib.pyplot as plt
from Research.feature.ft import FeatureAnalysis
import Research.utils.namespace as namespace
import Research.utils.normalization as norm
from Platform.database.mysql import MysqlAPI
from Platform.utils.persistence import convert_to_standard_daily_feature_csv, convert_to_standard_daily_feature_par
from Platform.config.mysql_info import FACTOR_LIB_MYSQL_TIO
import DataAPI
from os import walk
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from progressbar import ProgressBar
from tqdm import tqdm
import multiprocessing as mp
import sys
sys.path.append("/home/wuwenjun/jupyter_code/Shannon/AlphaNet/packages/")
import AlphaNet
from AlphaNet.Data import DataLoader
from AlphaNet.Models import AlphaNet_LSTM_V1
# import AlphaNet.packages.AlphaNet as AlphaNet
# from AlphaNet.packages.AlphaNet.Data import DataLoader
# from AlphaNet.packages.AlphaNet.Models import AlphaNet_LSTM_V1

# read_task
task_info = np.load("/home/ShareFolder/feature_platform/ti0/wuwenjun/#Factor_Description/Task.npy",allow_pickle=True).item()
device = task_info["Cuda"].pop()
task = task_info["Task"]
task_index = task[task["status"] == "waiting"].index[0]
Alpha_Name = task.loc[task_index,"Alpha_Name"]
start_date = task.loc[task_index,"start_date"]
end_date = task.loc[task_index,"end_date"]
sequence = task.loc[task_index,"sequence"]
LR = task.loc[task_index,"LR"]
epoch_num = task.loc[task_index,"epoch_num"]
feature_num = task.loc[task_index,"feature_num"]
t1 = (datetime.datetime.utcnow() + datetime.timedelta(hours=8)).strftime('%H:%M')
task.loc[task_index,"status"] = device + " : "+ t1 
np.save("/home/ShareFolder/feature_platform/ti0/wuwenjun/#Factor_Description/Task.npy",task_info)

# file path
model_path = "/home/wuwenjun/Alpha_Factor/" + Alpha_Name + "/" + "%s_%s" %(start_date,end_date) + "/"
if os.path.exists(model_path) == False:
    os.makedirs(model_path)
    print(model_path)
data_path = "/home/wuwenjun/Data/"

# write task
f = open(model_path + 'back_test.txt','w')
print("*"*100,end="\n"*3,file=f)
print("Alpha_Name: ",Alpha_Name,end = "\n",file=f)
print("start_date: ",start_date,end = "\n",file=f)
print("end_date: ",end_date,end = "\n",file=f)
print("sequence: ",sequence,end = "\n",file=f)
print("LR: ",LR,end = "\n",file=f)
print("epoch_num: ",epoch_num,end = "\n",file=f)
print("feature_num: ",feature_num,end = "\n"*3,file=f)
f.close()

# Train Loader

trainloader = DataLoader()
trainloader.load_data_from_file(alpha_name = Alpha_Name,end_date = start_date,data_path=data_path)
train_loader = trainloader.to_torch_DataLoader(sequence = sequence,shuffle=True)

# Model Loader

loss_function = nn.MSELoss()
model = AlphaNet_LSTM_V1(feature_num,sequence,64,attention = True)
optimizer = torch.optim.Adam(model.parameters(), lr=LR[0])
model_loader = AlphaNet.Model_Loader(model = model,device=device)
print(model_loader.model)

# Training
model = model_loader.fit_transform(train_loader,optimizer,loss_function,epoch_num[0],save_path = model_path)

for i in range(1,len(LR)):
    optimizer = torch.optim.Adam(model.parameters(), lr=LR[i])
    model = model_loader.transform(optimizer = optimizer, epoch_num = epoch_num[i], method="best")

# Test
testloader = DataLoader()
testloader.load_data_from_file(alpha_name = Alpha_Name,start_date = start_date,end_date = end_date,data_path = data_path)
test_loader = testloader.to_torch_DataLoader(sequence = sequence,shuffle=False)
pred = model_loader.pred(test_loader)

# convert to standard daily and back test
final = pd.concat([testloader.target.reset_index(),pred],axis=1)
final.rename(columns={0:Alpha_Name,'ticker': 'symbol'},inplace=True)
final.to_parquet(model_path + "result.par")
convert_to_standard_daily_feature_csv(Alpha_Name, final.drop("target",axis=1), output_path = r'/home/wuwenjun/factor_lib/ti0/wuwenjun')

# back test

pd.set_option('expand_frame_repr', False)
configs = namespace.load_namespace(r'/home/ShareFolder/lgc/Modules/Research/config/feature_bt_template')
FT = FeatureAnalysis(configs, feature_path=r"/home/wuwenjun/factor_lib")

FT.load_feature_from_file(Alpha_Name, "2019-01-01", end_date, universe='Investable',timedelta=None, transformer=norm.standard_scale)
FT.load_return_data()
FT.get_intersection_ic(feature_name=Alpha_Name, truncate_fold=None, method='spearman',period=('1d', '3d', '5d'))
ic_flag, trading_direction = FT.test_ic(Alpha_Name, verbose=False)
df, df_all = FT.get_ic_summary_by_month(num=6)

if trading_direction == -1:
    negative = True
else:
    negative = False


# save txt
f = open(model_path + 'back_test.txt','a')
print(df,end="\n"*3,file = f)
a = FT.get_top_return(Alpha_Name, negative= False, trade_type='long-only', transmission_rate=0.00025,
                        tax_rate=0.001, verbose=True,bt_price = "vwap",trade_period=6)
print(a[1],end = "\n"*3+"*"*100 ,file = f)
f.close()

# task
task_info = np.load("/home/ShareFolder/feature_platform/ti0/wuwenjun/#Factor_Description/Task.npy",allow_pickle=True).item()
task = task_info["Task"]
t2 = (datetime.datetime.utcnow() + datetime.timedelta(hours=8)).strftime('%H:%M | %m-%d')
task.loc[task_index,"status"] = "Finish: " + t1 + "_"+ t2
task_info["Cuda"].append(device)
np.save("/home/ShareFolder/feature_platform/ti0/wuwenjun/#Factor_Description/Task.npy",task_info)