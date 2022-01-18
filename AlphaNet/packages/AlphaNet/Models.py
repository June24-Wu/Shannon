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
    def __init__(self, factor_num, fully_connect_layer_neural):
        super(AlphaNet_LSTM_V1, self).__init__()
        self.fc1_neuron = factor_num  # 108
        self.fc2_neuron = fully_connect_layer_neural  # 32

        # Layer
        self.batch = torch.nn.BatchNorm1d(self.fc1_neuron)
        self.lstm = nn.LSTM(self.fc1_neuron, self.fc2_neuron, 5, batch_first=True, bidirectional=True)
        self.batch2 = torch.nn.BatchNorm1d(self.fc2_neuron *2)
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()
        self.out = nn.Linear(self.fc2_neuron * 2, 1)

    def forward(self, x):
        x = torch.transpose(x, 1, 2)
        x = self.batch(x)
        x = torch.transpose(x, 1, 2)
        r_out, (hn, cn) = self.lstm(x)  # hn.shape: torch.Size([4, 512, 30])
        r_out = r_out[:, -1]
        # hn = hn[-1, :, :]  # torch.Size([512, 30])
        r_out = self.batch2(r_out)  # torch.Size([512, 30])
        r_out = self.relu(r_out)
        r_out = self.dropout(r_out)
        y_pred = self.out(r_out)
        return y_pred