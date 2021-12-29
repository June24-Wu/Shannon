from os import walk
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from progressbar import ProgressBar
from tqdm import tqdm
import torch.utils.data as Data
import torch.optim as optim
from tqdm import tqdm
from torch.autograd import Variable
import time
import multiprocessing as mp
from os import walk
import matplotlib.pyplot as plt


def AlphaNet_Train_Test(Alpha_Name,start_time,feat_num):
    """
    Alpha_Name = "AlphaNet_Original_Input_5d_return"
    start_time = "2021-01-01"
    feat_num = 9
    """


    
    """Parameter"""
    batch_size = 1024
    workers = 32
    epoch_num = 30
    LR = 0.0001
    forecast_months = 6 


    path = '/home/wuwenjun/Data/' + Alpha_Name +'/'
    output_path = "/home/wuwenjun/Alpha_Factor/" + Alpha_Name + "/result/"
    model_dir = "/home/wuwenjun/Alpha_Factor/" + Alpha_Name + "/model/"
    if os.path.exists(output_path) == False:
        os.makedirs(output_path)

    time_list = []
    data_path = path + "Data/"
    dataframe_list = pd.DataFrame()
    for f, _, i in walk(data_path):
        for j in tqdm(i):
            time_list.append(j)
    time_list.sort()     
    for count,item in enumerate(time_list):
        if item.startswith(start_time):
            train_timestamp = time_list[:count]
            test_timestamp = time_list[count:count + forecast_months//3]
            break

    # Save Model
    model_path = (model_dir + test_timestamp[0].split("_")[0] + "_" + test_timestamp[-1].split("_")[-1]).replace(".par","/")
    if os.path.exists(model_path) == False:
        os.makedirs(model_path)

    print("Train:")
    display(train_timestamp)
    print("Test:")
    display(test_timestamp)

    """Train Test Split"""
    trainx , trainy , testx , testy = [] , [] , [],  []

    for train in tqdm(train_timestamp):
        df = pd.read_parquet(path+ "Final/" + train).set_index(["timestamp","ticker"])
        trainx.append(df.drop("target",axis=1))
        trainy.append(df['target'])
    trainx = pd.concat(trainx,axis=0)
    trainy = pd.concat(trainy,axis=0)


    for test in tqdm(test_timestamp):
        df = pd.read_parquet(path+ "Final/" + test).set_index(["timestamp","ticker"])
        testx.append(df.drop("target",axis=1))
        testy.append(df['target'])
    testx = pd.concat(testx,axis=0)
    testy = pd.concat(testy,axis=0)
    target_list = pd.DataFrame(testy.copy())

    trainx = torch.from_numpy(np.array(trainx))
    trainy = torch.from_numpy(np.array(trainy).reshape(-1,1))
    testx = torch.from_numpy(np.array(testx))
    testy = torch.from_numpy(np.array(testy).reshape(-1,1))
    print("trainx.shape: " , trainx.shape)
    print("trainy.shape: " , trainy.shape)
    print("testx.shape: " , testx.shape)
    print("testy.shape: " , testy.shape)


    train_dataset = Data.TensorDataset(trainx, trainy)
    test_dataset = Data.TensorDataset(testx, testy)
    train_loader = Data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=True
    )

    test_loader = Data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True
    )

    """AlphaNet"""
    class AlphaNet(nn.Module):
        def __init__(self, factor_num, fully_connect_layer_neural):
            super(AlphaNet, self).__init__()
            self.fc1_neuron = int((factor_num * (factor_num - 1)+ 4 * factor_num) * 3 * 2)
            self.fc2_neuron = fully_connect_layer_neural
            self.batch = torch.nn.BatchNorm1d(self.fc1_neuron)
            self.fc1 = torch.nn.Linear(self.fc1_neuron, self.fc2_neuron)
            self.dropout = nn.Dropout(0.3)
            self.relu = nn.ReLU()
            self.out = nn.Linear(self.fc2_neuron, 1)

        def forward(self, x):
            x = self.batch(x)
            x = self.fc1(x)
            x = self.relu(x)
            x = self.dropout(x)
            y_pred = self.out(x)
            return y_pred

    """Training"""
    alphanet = AlphaNet(feat_num, 30)
    alphanet = alphanet.cuda()
    print(alphanet)
    total_length = trainx.shape[0]
    LR = 0.0001
    loss_function = nn.MSELoss().cuda()
    optimizer = optim.RMSprop(alphanet.parameters(), lr=LR, alpha=0.9)
    epoch_num = 30
    loss_list = []

    for epoch in tqdm(range(epoch_num)):
        total_loss = 0
        for _, (inputs, outputs) in enumerate(train_loader):
            inputs = Variable(inputs).float().cuda()
            outputs = Variable(outputs).float().cuda()
            optimizer.zero_grad() # noticed:  the grad return to zero before starting the loop
            
            # forward + backward +update
            pred = alphanet(inputs)
            pred = pred.cuda()
            loss = loss_function(pred, outputs)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        total_loss = total_loss * batch_size / total_length
        print('Epoch: ', epoch + 1, ' loss: ', total_loss)
        loss_list.append(total_loss)

    # Save
    torch.save(alphanet,model_path + "model.tar")
    np.save(model_path + "loss.npy", loss_list)
    plt.plot(loss_list,color = 'r')
    plt.savefig(model_path+"loss.png")


    # 
    alphanet = alphanet.cpu()
    pred_list = []
    label_list = []
    for _, (data, label) in enumerate(test_loader):
        data = Variable(data).float()
        pred = alphanet(data)
        pred_list.extend(pred.tolist())
        label_list.extend(label.tolist())

    final = pd.DataFrame(pred_list)
    final = pd.concat([target_list.reset_index(),final],axis=1)
    final.rename(columns={0:Alpha_Name,'ticker': 'symbol'},inplace=True)
    display(final)
    final.to_parquet(output_path
                    + test_timestamp[0].split("_")[0] + "_" + test_timestamp[-1].split("_")[-1])
