import time
import datetime
import os
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
sys.path.append("/AlphaNet/packages/")
import AlphaNet
from AlphaNet.Data import concat_original_data , generate_shift_data

# read_task
task_info = np.load("/home/ShareFolder/feature_platform/ti0/wuwenjun/#Factor_Description/Task.npy",allow_pickle=True).item()
factor_info = task_info["Factor"]
if task_info["CPU"] == False:
    raise ValueError("No CPU is avaliable")
factor_index = factor_info[factor_info["status"] == "waiting"].index[0]
alpha_name = factor_info.loc[factor_index,"Alpha_Name"]
shift = factor_info.loc[factor_index,"shift"]
sequence = factor_info.loc[factor_index,"sequence"]
target = factor_info.loc[factor_index,"target"]
LR = factor_info.loc[factor_index,"LR"]
epoch_num = factor_info.loc[factor_index,"epoch_num"]
alpha_list = factor_info.loc[factor_index,"alpha_list"]
universe = factor_info.loc[factor_index,"universe"]
t1 = (datetime.datetime.utcnow() + datetime.timedelta(hours=8)).strftime('%H:%M')
factor_info.loc[factor_index,"status"] = "Running: " +t1
task_info["CPU"] = False
np.save("/home/ShareFolder/feature_platform/ti0/wuwenjun/#Factor_Description/Task.npy",task_info)
print(factor_info.loc[factor_index,:])

output_path = "/home/wuwenjun/Data/"

if alpha_name not in os.listdir(output_path):
    concat_original_data(alpha_name=alpha_name,alpha_list=alpha_list,output_path=output_path,universe = universe) # concat feature
generate_shift_data(alpha_name=alpha_name,shift=shift,sequence=sequence,target=target,data_path=output_path) # shift feature into the two dimention metrix

# Task Generation
task_info = np.load("/home/ShareFolder/feature_platform/ti0/wuwenjun/#Factor_Description/Task.npy",allow_pickle=True).item()
original = task_info["Task"]
# Timelist & Task
time_list = ["2019-01-01","2019-06-01","2020-01-01","2020-06-01","2021-01-01","2021-06-01"]
time_list2 = []
for i in range(len(time_list)-1):
    time_list2.append([time_list[i],time_list[i+1]])
time_list2 = pd.DataFrame(time_list2,columns=["start_date","end_date"])

task = pd.DataFrame([
                    alpha_name + "_Shift_%i_Sequence_%i_%s" % (shift,sequence,target),
                    sequence,LR,epoch_num,len(alpha_list),
                        ],index=["Alpha_Name","sequence","LR","epoch_num","feature_num"]).T
task['value']=1
time_list2['value']=1
task = pd.merge(task,time_list2,how='left',on='value')
del task['value']
task["status"] = "waiting"
task["description"] = [{
    "target" : target,
    "alpha_name" : alpha_name,
    "alpha_list" : alpha_list,
    "shift" : shift,
    "sequence" : sequence
} for i in range(len(task))]
task = pd.concat([original,task],axis=0)
task.reset_index(drop=True,inplace=True)
task.index.names = ["task_id"]
task_info["Task"] = task
factor_info = task_info["Factor"]
t2 = (datetime.datetime.utcnow() + datetime.timedelta(hours=8)).strftime('_%H:%M | %m-%d')
factor_info.loc[factor_index,"status"] = "Finish: " +t1+t2
task_info["CPU"] = True
np.save("/home/ShareFolder/feature_platform/ti0/wuwenjun/#Factor_Description/Task.npy",task_info)
np.save("/AlphaNet/Factor_Description/Task.npy", task_info)