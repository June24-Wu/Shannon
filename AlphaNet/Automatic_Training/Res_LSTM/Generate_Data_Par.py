import datetime
import os
import pandas as pd
import numpy as np
import sys
sys.path.append("/home/wuwenjun/jupyter_code/Shannon/AlphaNet/packages/")
import AlphaNet
from AlphaNet.Data import concat_original_data , generate_shift_data , DataLoader


alpha_name = input("Please Input the [Alpha_Name] that you want to load three dimention data: ")
split_day = input("Please Input the different moving days(split with ',' such as 10,5,20): ")
split_day = split_day.split(",")

# read_task
task_info = np.load("/home/ShareFolder/feature_platform/ti0/wuwenjun/#Factor_Description/Task.npy",allow_pickle=True).item()
factor_info = task_info["Factor"]
if task_info["CPU"] == False:
    raise ValueError("No CPU is avaliable")
factor_index = factor_info[factor_info["Alpha_Name"] == alpha_name].index
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

alpha_name_list = []
alpha_name_temp = alpha_name + "_" + split_day[0]
alpha_name_list.append(alpha_name_temp)
alpha_list_temp = [i + "_" + split_day[0] for i in alpha_list]
if alpha_name_temp not in os.listdir(output_path):
    concat_original_data(alpha_name=alpha_name_temp,alpha_list=alpha_list_temp,output_path=output_path,universe = universe)
generate_shift_data(alpha_name=alpha_name_temp,shift=shift,sequence=sequence,target=target,data_path=output_path)

for split in split_day[1:]:
    alpha_name_temp = alpha_name + "_" + split
    alpha_name_list.append(alpha_name_temp)
    alpha_list_temp = [i + "_" + split for i in alpha_list]
    if alpha_name_temp not in os.listdir(output_path):
        concat_original_data(alpha_name=alpha_name_temp, alpha_list=alpha_list_temp, output_path=output_path,
                             universe=universe)
    generate_shift_data(alpha_name=alpha_name_temp, shift=shift, sequence=sequence, target=None,
                        data_path=output_path)


final_df = []
for i in alpha_name_list:
    loader = DataLoader()
    loader.load_data_from_file(alpha_name="i",data_path=output_path)
    final_df.append(loader.feature_data)
final_df = pd.concat(final_df,axis=1)
final_df.dropna(axis=0,inplace=True)

AlphaNet.Data.convert_to_standard_daily_data_par(final_df,alpha_name,output_path)


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
task["status"] = "res_lstm"
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
np.save("/home/wuwenjun/jupyter_code/Shannon/AlphaNet/Factor_Description/Task.npy",task_info)