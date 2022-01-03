from os import walk
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

    torch.load()
    print('trainx size: ', trainx.size())
    print('trainy size: ', trainy.size())
    print('testx size: ', testx.size())
    print('testy size: ', testy.size())

    train_dataset = Data.TensorDataset(trainx, trainy)
    test_dataset = Data.TensorDataset(testx, testy)
    batch_size = 128
    train_loader = Data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )

    test_loader = Data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )

    alphanet = AlphaNet(feat_num, 30)
    print(alphanet)
    total_length = trainx.shape[0]
    LR = 0.000001
    loss_function = nn.MSELoss()
    optimizer = optim.RMSprop(alphanet.parameters(), lr=LR, alpha=0.9)
    epoch_num = 30

    for epoch in tqdm(range(epoch_num)):
        total_loss = 0
        for _, (data, label) in enumerate(train_loader):
            data = Variable(data).float()
            pred = alphanet(data)
            label = Variable(label).float()
            #         label = label.unsqueeze(1)
            loss = loss_function(pred, label)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()
        total_loss = total_loss * batch_size / total_length
        print('Epoch: ', epoch + 1, ' loss: ', total_loss)

    pred_list = []
    label_list = []

    for _, (data, label) in enumerate(test_loader):
        data = Variable(data).float()
        pred = alphanet(data)
        pred_list.extend(pred.tolist())
        label_list.extend(label.tolist())

    final = pd.concat([test_target, pd.DataFrame(pred_list)], axis=1)
    final = final[['timestamp', 'ticker', 0]]
    alpha_name = 'AlphaNet'
    final.rename(columns={0: alpha_name, 'ticker': 'symbol'}, inplace=True)
    final = final.reindex(columns=['symbol', 'timestamp', alpha_name])
    final.set_index(['symbol', 'timestamp']).to_csv(output_path + '%s_%s.csv' % (time_start, time_end))
    return None


if __name__ == '__main__':
time_list = [20190401, 20190630, 20191231, 20200601, 20201231, 20210630]
data_path = "/home/wuwenjun/Alpha_Factor/AlphaNetV1_Original_Input_1208/"
time_list = []
for f, _, i in walk(data_path + "trainx"):
    for j in tqdm(i):
        time_list.append(j)
time_list