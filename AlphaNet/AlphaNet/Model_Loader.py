import os
from os import walk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.optim as optim
from torch.autograd import Variable
from progressbar import ProgressBar
from tqdm import tqdm
import time
import multiprocessing as mp

def fit(dataloader,model,loss_function,optimizer,epoch_num,device,save_path = None):
    print("Learning Rate is :",optimizer.state_dict()['param_groups'][0]["lr"])
    loss_function = loss_function.to(device)
    loss_list = []

    min_loss = float("inf")
    for epoch in tqdm(range(epoch_num)):
        total_loss = 0
        for _, (inputs, outputs) in enumerate(dataloader):
            inputs = Variable(inputs).float().to(device)
            outputs = Variable(outputs).float().to(device)
            optimizer.zero_grad() # noticed:  the grad return to zero before starting the loop

            # forward + backward +update
            pred = model(inputs)
            pred = pred.cuda()
            loss = loss_function(pred, outputs)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        total_loss = total_loss * dataloader.batch_size / dataloader.dataset.tensors[0].shape[0]
        print('Epoch: ', epoch + 1, ' loss: ', total_loss)
        loss_list.append(total_loss)
        if total_loss < min_loss and save_path is not None:
            torch.save(model ,save_path + "best_model.tar")
        np.save(save_path + "loss.npy", loss_list)
    torch.save(model ,save_path + "model.tar")
    return model

