__all__ = ["Models", "DataProcessing"]


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from progressbar import ProgressBar
from tqdm import tqdm
import time
import multiprocessing as mp
from Research.feature.ft import FeatureAnalysis
import Research.utils.namespace as namespace
config_path = r'/home/ShareFolder/lgc/Modules/Research/config/feature_bt_template'
print('Loading the configuration from ' + config_path)
configs = namespace.load_namespace(config_path)


class Model_Loader(object):
    def __init__(self, model, device=None):
        if device == None:
            raise ValueError(
                r"please indicate device by running device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')")
        self.device = device
        self.model = model.to(self.device)
        self.best_model = model.to(self.device)
        self.optimizer = None
        self.loss_list = []
        self.min_loss = float("inf")
        self.dataloader = None
        self.loss_function = None
        self.save_path = None

    def fit_transform(self, dataloader, optimizer, loss_function, epoch_num, save_path=None):
        self.dataloader = dataloader
        self.loss_function = loss_function.to(self.device)
        self.save_path = save_path
        self.optimizer = optimizer
        print("Learning Rate is :", self.optimizer.state_dict()['param_groups'][0]["lr"])

        for epoch in tqdm(range(epoch_num)):
            total_loss = 0
            for _, (inputs, outputs) in enumerate(dataloader):
                inputs = Variable(inputs).float().to(self.device)
                outputs = Variable(outputs).float().to(self.device)
                self.optimizer.zero_grad()  # noticed:  the grad return to zero before starting the loop

                # forward + backward +update
                pred = self.model(inputs)
                pred = pred.to(self.device)
                loss = self.loss_function(pred, outputs)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
            total_loss = total_loss * dataloader.batch_size / dataloader.dataset.tensors[0].shape[0]
            print('Epoch: ', epoch + 1, ' loss: ', total_loss)
            self.loss_list.append(total_loss)
            if save_path is not None:
                np.save(save_path + "loss.npy", self.loss_list)
                if total_loss < self.min_loss:
                    self.best_model = self.model
                    torch.save(self.model, save_path + "best_model.tar")
                    self.min_loss = total_loss
        if save_path is not None:
            torch.save(self.model, save_path + "model.tar")
            plt.plot(self.loss_list, color='r')
            plt.savefig(save_path + "loss.png")
        return self.model

    def transform(self, optimizer, epoch_num, method=None):
        if method == "best":
            self.model = self.best_model
        return self.fit_transform(self.dataloader, optimizer, self.loss_function, epoch_num, self.save_path)

    def pred(self, dataloader):
        pred_list = []
        label_list = []
        for _, (data, label) in enumerate(dataloader):
            data = Variable(data).float().to(self.device)
            pred = self.model(data).to(self.device)
            pred_list.extend(pred.tolist())
            label_list.extend(label.tolist())
        self.testy_pred = pd.DataFrame(pred_list)
        self.testy = pd.DataFrame(label_list)
        return self.testy_pred
