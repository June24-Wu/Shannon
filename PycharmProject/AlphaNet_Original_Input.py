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
from multiprocessing import Process



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

        feat_cat = torch.cat([bc1, bc2, bc3, bc4, bc5, bc6], axis=1)  # ÌØÕ÷¾ÛºÏ
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
        # ¹¹½¨²½³¤ÁÐ±í£¬Èç¹ûÊý¾Ý³¤¶È²»ÄÜÕû³ý£¬ÔòÈ¡Ê£ÏÂ³¤¶È£¬Èç¹ûÊ£ÏÂ³¤¶ÈÐ¡ÓÚ5£¬ÔòÓëÉÏÒ»²½½áºÏÒ»Æð
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
        '''¼ÆËã4Î¬Êý¾ÝµÄÐ­·½²î'''
        '''data:[N,C,H,W],,W:price length,N:batch size'''
        l = []
        # ¼ÆËãµÄ¹ý³ÌÖÐÎñ±Ø±£³Ökeepdims=True
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
        '''¼ÆËã4Î¬Êý¾ÝµÄÏà¹ØÏµÊý'''
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
            del std1, std2  # ÊÍ·ÅÄÚ´æ
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
    def __init__(self,data,stride):
        if len(data.shape)!=4:
            raise Exception('Input data dimensions should be [N,C,H,W]')
        self.data = data.detach().numpy()
        self.stride = stride
        self.data_length = data.shape[3]
        self.feat_num = data.shape[2] # 9
        self.step_list = self.generate_Step_List(self.data_length,self.stride)
        self.extracted_data = self.Extraction(self.data,self.feat_num,self.stride)
    def Extraction(self,data,feat_num,stride):
        print("------Start Pooling------")
        # Pooling
        ts_max = self.ts_pool(data,self.stride,self.feat_num,self.step_list,method = 'max')
        ts_max = nn.BatchNorm1d(self.feat_num,affine = True)(ts_max)
        ts_min = self.ts_pool(data ,self.stride,self.feat_num,self.step_list,method = 'min')
        ts_min = nn.BatchNorm1d(self.feat_num,affine = True)(ts_min)
        ts_mean = self.ts_pool(data ,self.stride,self.feat_num,self.step_list,method = 'mean')
        ts_mean = nn.BatchNorm1d(self.feat_num,affine = True)(ts_mean)
        data_pool = torch.cat([ts_max,ts_min,ts_mean],axis = 1)
        data_pool = data_pool.flatten(start_dim = 1)
        print("Pooling shape: ",data_pool.shape )
        return data_pool
    def generate_Step_List(self,data_length,stride):
        # ¹¹½¨²½³¤ÁÐ±í£¬Èç¹ûÊý¾Ý³¤¶È²»ÄÜÕû³ý£¬ÔòÈ¡Ê£ÏÂ³¤¶È£¬Èç¹ûÊ£ÏÂ³¤¶ÈÐ¡ÓÚ5£¬ÔòÓëÉÏÒ»²½½áºÏÒ»Æð
        if data_length % stride == 0:
            step_list = list(range(0,data_length+stride,stride))
        elif data_length % stride<=5:
            mod = data_length % stride
            step_list = list(range(0,data_length-stride,stride))+[data_length]
        else:
            mod = data_length % stride
            step_list = list(range(0,data_length+stride-mod,stride))+[data_length]
        return step_list
    def ts_pool(self,data,stride,feat_num,step_list,method):
        if type(data) == torch.Tensor:
            data = data.detach().numpy()
        if data.shape[-1] <= stride:
            step_list = [0,data.shape[-1]]
        if len(data.shape)!=4:
            raise Exception('Input data dimensions should be [N,C,H,W]')
        l = []
        for i in tqdm(range(len(step_list)-1)):
            start = step_list[i]
            end = step_list[i+1]
            if method == 'max':
                sub_data1 = data[:,:,:,start:end].max(axis = 3,keepdims = True)
            if method == 'min':
                sub_data1 = data[:,:,:,start:end].min(axis = 3,keepdims = True)
            if method == 'mean':
                sub_data1 = data[:,:,:,start:end].mean(axis = 3,keepdims = True)
            l.append(sub_data1)
        try:
            pool_data = np.squeeze(np.array(l)).transpose(1,2,0).reshape(-1,feat_num,len(step_list) - 1)
        except:
            pool_data = np.squeeze(np.array(l)).reshape(-1,feat_num,len(step_list) - 1)
        return torch.from_numpy(pool_data)

class AlphaNet(nn.Module):
    def __init__(self,factor_num,fully_connect_layer_neural):
        # super ¸¸Àà£¬µ÷ÓÃ¸¸ÀàµÄ¹¹Ôì£¬ÕâÒ»²½±ØÐëÓÐ
        # µÚÒ»¸ö²ÎÊýÎª¶¨ÒåÀàµÄÃû³Æ£¬µÚ¶þ¸öÎªself
        super(AlphaNet,self).__init__()
        self.fc1_neuron = (factor_num * (factor_num -1) + 4 * factor_num)* 3
        self.fc2_neuron = fully_connect_layer_neural
        self.fc1 = torch.nn.Linear(self.fc1_neuron,self.fc2_neuron)
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()
        self.out = nn.Linear(self.fc2_neuron,1)
    def forward(self,x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        y_pred = self.out(x)
        return y_pred

def test(time_start, time_end):
    train_frame = dataframe_list[dataframe_list['timestamp'] < pd.to_datetime(time_start)]
    test_frame = dataframe_list[(dataframe_list['timestamp'] > pd.to_datetime(time_start))
                                & (dataframe_list['timestamp'] < pd.to_datetime(time_end))]

    trainx , trainy = [] , []
    for ticker in tqdm(train_frame['ticker'].drop_duplicates()):
        one_data = train_frame[train_frame['ticker'] == ticker]
        one_data = one_data.set_index(['timestamp','ticker'])
        array = np.array(one_data)
        for i in range(0,array.shape[0] - day ,3): # ÆäÖÐ3 ´ú±íÈ¡ÊýµÄ²½³¤£¬ex.Ã¿Á½ÌìÈ¡Ò»´ÎÊý£¬²½³¤Îª3
            trainx.append(array[i:i+day,:-1].T)
            trainy.append(array[i+day-1][-1])
    trainx  , trainy = np.array(trainx) , np.array(trainy).reshape(-1,1) # x = (153, 9, 30) , y = (153,1)
    trainx = trainx.reshape(trainx.shape[0],1,trainx.shape[1],trainx.shape[2]) # x = (153, 1, 9, 30)
    feat_num = trainx.shape[2]
    del train_frame
    print("trainx.shape: ",trainx.shape)
    print("trainy.shape: ",trainy.shape)

    testx,testy = [],[]
    test_target = pd.DataFrame()
    for ticker in tqdm(test_frame['ticker'].drop_duplicates()):
        one_data = test_frame[test_frame['ticker'] == ticker]
        one_data = one_data.set_index(['timestamp','ticker'])
        array = np.array(one_data)
        one_data = one_data.reset_index()
        for i in range(0,array.shape[0] - day ,3): # ÆäÖÐ3 ´ú±íÈ¡ÊýµÄ²½³¤£¬ex.Ã¿Á½ÌìÈ¡Ò»´ÎÊý£¬²½³¤Îª3
            testx.append(array[i:i+day,:-1].T)
            testy.append(array[i+day-1][-1])
            temp = pd.DataFrame(one_data.iloc[i+day-1,:]).T
            test_target = pd.concat([test_target,temp[['timestamp','ticker',target]]],axis=0)
    #         test_target = pd.concat([test_target,pd.DataFrame(one_data.iloc[i+day-1,-1])])
    testx  , testy = np.array(testx) , np.array(testy).reshape(-1,1) # x = (153, 9, 30) , y = (153,1)
    testx = testx.reshape(testx.shape[0],1,testx.shape[1],testx.shape[2]) # x = (153, 1, 9, 30)
    del test_frame
    print("testx.shape: ",testx.shape)
    print("testy.shape: ",testy.shape)
    test_target.reset_index(inplace = True,drop = True)


    """Convolutional """
    convolutional = Convolutional(trainx,10)
    feat_cat = convolutional.extracted_data
    pooling = Pooling(feat_cat,3)
    trainx = pooling.extracted_data.detach().numpy()
    print("trainx.shape : ",trainx.shape)
    print("trainy.shape : ",trainy.shape)

    convolutional = Convolutional(testx,10)
    feat_cat = convolutional.extracted_data
    pooling = Pooling(feat_cat,3)
    testx = pooling.extracted_data.detach().numpy()
    print("testx.shape : ",testx.shape)
    print("testy.shape : ",testy.shape)

    trainx , trainy  , testx , testy = torch.from_numpy(trainx) , torch.from_numpy(trainy) , torch.from_numpy(testx) , torch.from_numpy(testy)
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


    alphanet = AlphaNet(feat_num,30)
    print(alphanet)
    total_length = trainx.shape[0]
    LR = 0.000001
    loss_function = nn.MSELoss()
    optimizer = optim.RMSprop(alphanet.parameters(), lr=LR, alpha=0.9)
    epoch_num = 30


    for epoch in tqdm(range(epoch_num)):
        total_loss = 0
        for _,(data, label) in enumerate(train_loader):
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
        print('Epoch: ', epoch+1, ' loss: ', total_loss)

    pred_list = []
    label_list = []

    for _, (data, label) in enumerate(test_loader):
        data = Variable(data).float()
        pred = alphanet(data)
        pred_list.extend(pred.tolist())
        label_list.extend(label.tolist())
    # for i in range(1000):
    #     print("pred: ", pred_list[i][0])
    #     print("true: ",label_list[i][0])
    #     print("---------------------")
    final = pd.concat([test_target,pd.DataFrame(pred_list)],axis=1)
    final = final[['timestamp','ticker',0]]
    alpha_name = 'AlphaNet'
    final.rename(columns={ 0 : alpha_name ,'ticker': 'symbol'}, inplace=True)
    final = final.reindex(columns=['symbol', 'timestamp', alpha_name])
    final.set_index(['symbol','timestamp']).to_csv('/home/wuwenjun/Alpha_Factor/AlphaNetV1_Original_Input/%s_%s.csv' % (time_start,time_end))
    return None


if __name__ == '__main__':
    time_list = [20190401,20190630,20191231,20200601,20201231,20210630]
    dataframe_list = pd.read_csv('/home/wuwenjun/Data/AlphaNet_Original_Input.csv')
    dataframe_list['timestamp'] = pd.to_datetime(dataframe_list['timestamp'])
    day = 30
    stride = 10
    target = '5d_ret'

    # 多进程
    start_time = time.time()
    for i in range(len(time_list)-1):
        start_time = time_list[i]
        end_time = time_list[i+1]
        p = Process(target=test, args=(start_time,end_time))
        p.start()
    multi_end = time.time()
    print('\nMulti process cost time:', multi_end - start_time)








