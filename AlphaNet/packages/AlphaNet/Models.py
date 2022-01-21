import torch
import torch.nn as nn

class AlphaNet_V1(nn.Module):
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

class AlphaNet_LSTM_V1(nn.Module):
    def __init__(self, factor_num,sequence,fully_connect_layer_neural):
        super(AlphaNet_LSTM_V1, self).__init__()
        self.factor_num = factor_num  # 108
        self.sequence = sequence
        self.fc2_neuron = fully_connect_layer_neural  # 32

        # Layer
        self.batch = torch.nn.BatchNorm1d(self.sequence * self.factor_num)
        self.lstm = nn.LSTM(self.factor_num, self.fc2_neuron, 3, batch_first=True, bidirectional=True,dropout = 0.2)
        self.lstm2 = nn.LSTM(int(self.fc2_neuron *2), int(self.fc2_neuron/2), 3, batch_first=True, bidirectional=True,dropout = 0.2)
        self.batch2 = torch.nn.BatchNorm1d(int(self.fc2_neuron *2))
        self.batch3 = torch.nn.BatchNorm1d(self.fc2_neuron)
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()
        self.LeakyReLU = nn.LeakyReLU()
        self.out = nn.Linear(self.fc2_neuron, 1)

    def forward(self, x):
        x = x.reshape(x.shape[0],-1).float()
        x = self.batch(x)
        x = x.reshape(x.shape[0],self.sequence,self.factor_num)

        x, _ = self.lstm(x)  # x.shape: torch.Size([6182, 10, 128])
        x = self.LeakyReLU(x)
        x = torch.transpose(x,1,2) # x.shape: torch.Size([6182, 128, 10])
        x = self.batch2(x)
        x = torch.transpose(x,1,2)


        x, _ = self.lstm2(x) # torch.Size([6182, 10, 64])

        x = x[:, -1] # torch.Size([6182, 64])
        x = self.batch3(x)  # torch.Size([6182, 64])
        x = self.relu(x)
        x = self.dropout(x)
        y_pred = self.out(x)
        return y_pred