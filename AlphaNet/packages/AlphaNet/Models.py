import torch
import torch.nn as nn

class AlphaNet_LSTM_V1(nn.Module):
    def __init__(self, factor_num, fully_connect_layer_neural):
        super(AlphaNet_LSTM_V1, self).__init__()
        self.fc1_neuron = factor_num  # 108
        self.fc2_neuron = fully_connect_layer_neural  # 30

        # Layer
        self.batch = torch.nn.BatchNorm1d(self.fc1_neuron)
        self.lstm = nn.LSTM(self.fc1_neuron, self.fc2_neuron, 2, batch_first=True, bidirectional=True)
        self.batch2 = torch.nn.BatchNorm1d(self.fc2_neuron)
        self.dropout = nn.Dropout(0.3)
        #         self.relu = nn.ReLU()
        self.out = nn.Linear(self.fc2_neuron, 1)

    def forward(self, x):
        x = self.batch(x)
        x = torch.transpose(x, 1, 2)
        _, (hn, cn) = self.lstm(x)  # hn.shape: torch.Size([4, 512, 30])
        hn = hn[-1, :, :]  # torch.Size([512, 30])
        hn = self.batch2(hn)  # torch.Size([512, 30])
        #         hn = self.relu(hn)
        hn = self.dropout(hn)
        y_pred = self.out(hn)
        return y_pred