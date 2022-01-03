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


class Convolutional(object):
    def __init__(self, data, stride):
        if len(data.shape) != 4:
            raise Exception('Input data dimensions should be [N,C,H,W]')
        self.data = np.array(data)
        self.stride = stride
        self.data_length = data.shape[3]
        self.feat_num = data.shape[2]  # 9
        self.num, self.num_rev = self.generate_Num_and_ReversedNum(self.feat_num)
        self.conv_feat = len(self.num)
        self.step_list = self.generate_Step_List(self.data_length, self.stride)
        self.extracted_data = self.Extraction(self.data, self.feat_num, self.conv_feat, self.stride)

    def Extraction(self, data, feat_num, conv_feat, stride):
        print("------Start Extraction------")
        batch = nn.BatchNorm1d(conv_feat, affine=True)
        batch2 = nn.BatchNorm1d(feat_num, affine=True)
        conv1 = self.ts_cov4d(self.data, self.stride, self.num, self.num_rev, self.step_list)
        conv2 = self.ts_corr4d(self.data, self.stride, self.num, self.num_rev, self.step_list, conv1)
        conv2 = torch.tanh(conv2)
        bc1 = batch(conv1.to(torch.float))
        bc2 = batch(conv2.to(torch.float))
        conv3 = self.ts_stddev4d(self.data, self.stride, self.feat_num, self.step_list).to(torch.float)
        bc3 = batch2(conv3)
        conv4 = self.ts_zscore(self.data, self.stride, self.feat_num, self.step_list).to(torch.float)
        bc4 = batch2(conv4)
        conv5 = self.ts_return(self.data, self.stride, self.feat_num, self.step_list).to(torch.float)
        bc5 = batch2(conv5)
        conv6 = self.ts_decaylinear(self.data, self.stride, self.feat_num, self.step_list).to(torch.float)
        bc6 = batch2(conv6)

        feat_cat = torch.cat([bc1, bc2, bc3, bc4, bc5, bc6], axis=1)  # ¨¬??¡Â??o?
        shape = feat_cat.shape
        feat_cat = feat_cat.reshape(shape[0], 1, shape[1], shape[2])
        print("Convolutional shape: ", feat_cat.shape)
        return feat_cat

    def generateC(self, l1):
        if len(l1) == 1:
            return []
        v = [[l1[0], i] for i in l1[1:]]
        l1 = l1[1:]
        return v + self.generateC(l1)

    def generate_Num_and_ReversedNum(self, feat_nums):
        list1 = list(range(feat_nums))
        num = self.generateC(list1)
        num_rev = []
        for l in num:
            l1 = l.copy()
            l1.reverse()
            num_rev.append(l1)
        return num, num_rev

    def generate_Step_List(self, data_length, stride):
        # 11?¡§2?3¡è¨¢D¡À¨ª¡ê?¨¨?1?¨ºy?Y3¡è?¨¨2??¨¹??3y¡ê??¨°¨¨?¨º¡ê??3¡è?¨¨¡ê?¨¨?1?¨º¡ê??3¡è?¨¨D?¨®¨²5¡ê??¨°¨®?¨¦?¨°?2??¨¢o?¨°??e
        if data_length % stride == 0:
            step_list = list(range(0, data_length + stride, stride))
        elif data_length % stride <= 5:
            mod = data_length % stride
            step_list = list(range(0, data_length - stride, stride)) + [data_length]
        else:
            mod = data_length % stride
            step_list = list(range(0, data_length + stride - mod, stride)) + [data_length]
        return step_list

    """ Main Extraction"""

    def ts_cov4d(self, data, stride, num, num_rev, step_list):
        '''????4??¨ºy?Y¦Ì?D-¡¤?2?'''
        '''data:[N,C,H,W],,W:price length,N:batch size'''
        l = []
        # ????¦Ì?1y3¨¬?D??¡À?¡À¡ê3?keepdims=True
        for i in tqdm(range(len(step_list) - 1)):
            start = step_list[i]
            end = step_list[i + 1]
            sub_data1 = data[:, :, num, start:end]  # (2000, 1, 36, 2, 10)
            sub_data2 = data[:, :, num_rev, start:end]
            mean1 = sub_data1.mean(axis=4, keepdims=True)  # (2000, 1, 36, 2, 1)
            mean2 = sub_data2.mean(axis=4, keepdims=True)
            spread1 = sub_data1 - mean1  # (2000, 1, 36, 2, 10)
            spread2 = sub_data2 - mean2
            cov = ((spread1 * spread2).sum(axis=4, keepdims=True) / (sub_data1.shape[4] - 1)).mean(axis=3,
                                                                                                   keepdims=True)  # (2000, 1, 36, 1, 1)
            l.append(cov)
        corr = np.squeeze(np.array(l)).transpose(1, 2, 0).reshape(-1, self.conv_feat,
                                                                  len(step_list) - 1)  # (2000, 1, 36, 3)
        final = torch.from_numpy(corr)
        print("------Finished ts_cov4d----output shape: ", final.shape)
        return final

    def ts_corr4d(self, data, stride, num, num_rev, step_list, cov):
        '''????4??¨ºy?Y¦Ì??¨¤1??¦Ì¨ºy'''
        '''data:[N,C,H,W],,W:price length,N:batch size'''
        l = []
        for i in tqdm(range(len(step_list) - 1)):
            start = step_list[i]
            end = step_list[i + 1]
            sub_data1 = data[:, :, num, start:end]
            sub_data2 = data[:, :, num_rev, start:end]
            std1 = sub_data1.std(axis=4, keepdims=True)
            std2 = sub_data2.std(axis=4, keepdims=True)
            std = (std1 * std2).mean(axis=3, keepdims=True)
            del std1, std2  # ¨º¨ª¡¤??¨²¡ä?
            l.append(std)
        std = np.squeeze(np.array(l)).transpose(1, 2, 0).reshape(-1, self.conv_feat, len(step_list) - 1)
        std[std == 0] = 1e-9
        fct = (sub_data1.shape[4] - 1) / sub_data1.shape[4]
        final = cov / torch.from_numpy(std) * fct
        del fct, std
        print("------Finished ts_corr4d----output shape: ", final.shape)
        return final

    def ts_stddev4d(self, data, stride, feat_num, step_list):
        if len(data.shape) != 4:
            raise Exception('Input data dimensions should be [N,C,H,W]')
        l = []
        for i in tqdm(range(len(step_list) - 1)):
            start = step_list[i]
            end = step_list[i + 1]
            sub_data1 = data[:, :, :, start:end]
            std1 = sub_data1.std(axis=3, keepdims=True)
            l.append(std1)
            del std1
        std = np.squeeze(np.array(l)).transpose(1, 2, 0).reshape(-1, feat_num, len(step_list) - 1)
        print("------Finished ts_stddev4d----output shape: ", torch.from_numpy(std).shape)
        return torch.from_numpy(std)

    def ts_zscore(self, data, stride, feat_num, step_list):
        if len(data.shape) != 4:
            raise Exception('Input data dimensions should be [N,C,H,W]')
        l = []
        for i in tqdm(range(len(step_list) - 1)):
            start = step_list[i]
            end = step_list[i + 1]
            sub_data1 = data[:, :, :, start:end]
            mean = sub_data1.mean(axis=3, keepdims=True)
            std = sub_data1.std(axis=3, keepdims=True)
            std[std == 0] = 1e-9
            z_score = mean / std
            l.append(z_score)
        z_score = np.squeeze(np.array(l)).transpose(1, 2, 0).reshape(-1, feat_num, len(step_list) - 1)
        #         z_score[z_score >= 6] = 6
        print("------Finished ts_zscore----output shape: ", torch.from_numpy(z_score).shape)
        return torch.from_numpy(z_score)

    def ts_return(self, data, stride, feat_num, step_list):
        if len(data.shape) != 4:
            raise Exception('Input data dimensions should be [N,C,H,W]')
        data[data == 0] = 1e-9
        l = []
        for i in tqdm(range(len(step_list) - 1)):
            start = step_list[i]
            end = step_list[i + 1]
            sub_data1 = data[:, :, :, start:end]
            ret = sub_data1[:, :, :, -1] / sub_data1[:, :, :, 0] - 1
            l.append(ret)
        z_data = np.squeeze(np.array(l)).transpose(1, 2, 0).reshape(-1, feat_num, len(step_list) - 1)
        z_data[z_data > 1] = 1
        print("------Finished ts_return----output shape: ", torch.from_numpy(z_data).shape)
        return torch.from_numpy(z_data)

    def ts_decaylinear(self, data, stride, feat_num, step_list):
        if len(data.shape) != 4:
            raise Exception('Input data dimensions should be [N,C,H,W]')
        l = []
        for i in tqdm(range(len(step_list) - 1)):
            start = step_list[i]
            end = step_list[i + 1]
            time_spread = end - start
            weight = np.arange(1, time_spread + 1)
            weight = weight / (weight.sum())
            sub_data1 = (data[:, :, :, start:end] * weight).mean(axis=3, keepdims=True)
            l.append(sub_data1)
        decay_data = np.squeeze(np.array(l)).transpose(1, 2, 0).reshape(-1, feat_num, len(step_list) - 1)
        final = torch.from_numpy(decay_data)
        print("------Finished ts_decaylinear----output shape: ", final.shape)
        return final


class Pooling(object):
    def __init__(self, data, stride):
        if len(data.shape) != 4:
            raise Exception('Input data dimensions should be [N,C,H,W]')
        self.data = data.detach().numpy()
        self.stride = stride
        self.data_length = data.shape[3]
        self.feat_num = data.shape[2]  # 9
        self.step_list = self.generate_Step_List(self.data_length, self.stride)
        self.extracted_data = self.Extraction(self.data, self.feat_num, self.stride)

    def Extraction(self, data, feat_num, stride):
        print("------Start Pooling------")
        # Pooling
        ts_max = self.ts_pool(data, self.stride, self.feat_num, self.step_list, method='max')
        ts_max = nn.BatchNorm1d(self.feat_num, affine=True)(ts_max)
        ts_min = self.ts_pool(data, self.stride, self.feat_num, self.step_list, method='min')
        ts_min = nn.BatchNorm1d(self.feat_num, affine=True)(ts_min)
        ts_mean = self.ts_pool(data, self.stride, self.feat_num, self.step_list, method='mean')
        ts_mean = nn.BatchNorm1d(self.feat_num, affine=True)(ts_mean)
        data_pool = torch.cat([ts_max, ts_min, ts_mean], axis=1)
        data_pool = data_pool.flatten(start_dim=1)
        print("Pooling shape: ", data_pool.shape)
        return data_pool

    def generate_Step_List(self, data_length, stride):
        # 11?¡§2?3¡è¨¢D¡À¨ª¡ê?¨¨?1?¨ºy?Y3¡è?¨¨2??¨¹??3y¡ê??¨°¨¨?¨º¡ê??3¡è?¨¨¡ê?¨¨?1?¨º¡ê??3¡è?¨¨D?¨®¨²5¡ê??¨°¨®?¨¦?¨°?2??¨¢o?¨°??e
        if data_length % stride == 0:
            step_list = list(range(0, data_length + stride, stride))
        elif data_length % stride <= 5:
            mod = data_length % stride
            step_list = list(range(0, data_length - stride, stride)) + [data_length]
        else:
            mod = data_length % stride
            step_list = list(range(0, data_length + stride - mod, stride)) + [data_length]
        return step_list

    def ts_pool(self, data, stride, feat_num, step_list, method):
        if type(data) == torch.Tensor:
            data = data.detach().numpy()
        if data.shape[-1] <= stride:
            step_list = [0, data.shape[-1]]
        if len(data.shape) != 4:
            raise Exception('Input data dimensions should be [N,C,H,W]')
        l = []
        for i in tqdm(range(len(step_list) - 1)):
            start = step_list[i]
            end = step_list[i + 1]
            if method == 'max':
                sub_data1 = data[:, :, :, start:end].max(axis=3, keepdims=True)
            if method == 'min':
                sub_data1 = data[:, :, :, start:end].min(axis=3, keepdims=True)
            if method == 'mean':
                sub_data1 = data[:, :, :, start:end].mean(axis=3, keepdims=True)
            l.append(sub_data1)
        try:
            pool_data = np.squeeze(np.array(l)).transpose(1, 2, 0).reshape(-1, feat_num, len(step_list) - 1)
        except:
            pool_data = np.squeeze(np.array(l)).reshape(-1, feat_num, len(step_list) - 1)
        return torch.from_numpy(pool_data)


class AlphaNet(nn.Module):
    def __init__(self, factor_num, fully_connect_layer_neural):
        # super ??¨¤¨¤¡ê?¦Ì¡Â¨®???¨¤¨¤¦Ì?11?¨¬¡ê??a¨°?2?¡À?D?¨®D
        # ¦Ì¨²¨°???2?¨ºy?a?¡§¨°?¨¤¨¤¦Ì???3?¡ê?¦Ì¨²?t???aself
        super(AlphaNet, self).__init__()
        self.fc1_neuron = (factor_num * (factor_num - 1) + 4 * factor_num) * 3
        self.fc2_neuron = fully_connect_layer_neural
        self.fc1 = torch.nn.Linear(self.fc1_neuron, self.fc2_neuron)
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()
        self.out = nn.Linear(self.fc2_neuron, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        y_pred = self.out(x)
        return y_pred


def get_train_data(time_start, time_end):
    train_frame = dataframe_list[dataframe_list['timestamp'] < pd.to_datetime(str(time_start))]
    train_frame.set_index(["timestamp", "ticker"], inplace=True)

    # Train X and Train Y
    trainx, trainy = [], []
    trainx = np.array(train_frame.drop("target", axis=1))
    trainx = trainx.reshape(trainx.shape[0], 1, day, -1)  # x = (153,1,30,9)
    trainx = trainx.transpose(0, 1, 3, 2)  # x = (2555577, 1, 9, 30)
    trainy = np.array(train_frame['target']).reshape(-1, 1)  # x = (153,1,9,30)

    feat_num = trainx.shape[2]  # 9
    del train_frame
    print("trainx.shape: ", trainx.shape)
    print("trainy.shape: ", trainy.shape)
    return trainx, trainy, feat_num


def get_test_data(time_start, time_end):
    test_frame = dataframe_list[(dataframe_list['timestamp'] >= pd.to_datetime(str(time_start)))
                                & (dataframe_list['timestamp'] < pd.to_datetime(str(time_end)))]
    test_frame.set_index(["timestamp", "ticker"], inplace=True)

    # Test X and Test Y

    test_target = pd.DataFrame(test_frame['target'])
    testy = np.array(test_target).reshape(-1, 1)

    testx = np.array(test_frame.drop("target", axis=1))
    testx = testx.reshape(testx.shape[0], 1, day, -1)  # x = (153,1,30,9)
    testx = testx.transpose(0, 1, 3, 2)  # x = (2555577, 1, 9, 30)
    del test_frame

    print("testx.shape: ", testx.shape)
    print("testy.shape: ", testy.shape)
    test_target.reset_index(inplace=True)
    return testx, testy, test_target


def main(time_start, time_end):
    trainx, trainy, feat_num = get_train_data(time_start, time_end)
    testx, testy, test_target = get_test_data(time_start, time_end)

    """Convolutional """
    convolutional = Convolutional(trainx, 10)
    feat_cat = convolutional.extracted_data
    pooling = Pooling(feat_cat, 3)
    trainx = pooling.extracted_data.detach().numpy()
    print("trainx.shape : ", trainx.shape)
    print("trainy.shape : ", trainy.shape)

    convolutional = Convolutional(testx, 10)
    feat_cat = convolutional.extracted_data
    pooling = Pooling(feat_cat, 3)
    testx = pooling.extracted_data.detach().numpy()
    print("testx.shape : ", testx.shape)
    print("testy.shape : ", testy.shape)

    trainx, trainy, testx, testy = torch.from_numpy(trainx), torch.from_numpy(trainy), torch.from_numpy(
        testx), torch.from_numpy(testy)
    torch.save(trainx, output_path + "/trainx/" + '%s_%s.pt' % (time_start, time_end))
    torch.save(trainy, output_path + "/trainy/" + '%s_%s.pt' % (time_start, time_end))
    torch.save(testx, output_path + "/testx/" + '%s_%s.pt' % (time_start, time_end))
    torch.save(testy, output_path + "/testy/" + '%s_%s.pt' % (time_start, time_end))
    test_target.to_csv(output_path + "/test_target/" + '%s_%s.csv' % (time_start, time_end))
    return None
if __name__ == '__main__':
    output_path = "/home/wuwenjun/Alpha_Factor/AlphaNetV1_Original_Input_1208/"
    time_list = [20190101,20190630,20200101,20200630,20210101,20210630]
    day = 30
    stride = 10

    # Read Data
    data_path = "/home/wuwenjun/Data/AlphaNet_Original_Input/"
    dataframe_list = pd.DataFrame()
    for f, _, i in walk(data_path):
        for j in tqdm(i):
            dataframe_list = pd.concat([dataframe_list, pd.read_parquet(f + j)], axis=0)
    dataframe_list['timestamp'] = pd.to_datetime(dataframe_list['timestamp'])
    t1 = time.time()
    pool = mp.Pool(2)
    for i in range(len(time_list) - 1):
        start_time = pd.to_datetime(str(time_list[i]))
        end_time = pd.to_datetime(str(time_list[i + 1]))
        pool.apply_async(main, args=(start_time, end_time))
    pool.close()
    pool.join()
    t2 = time.time()
    print("running time", int(t2 - t1))








