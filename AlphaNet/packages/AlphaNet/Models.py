import torch
import torch.nn as nn

class AlphaNet_V1(nn.Module):
    def __init__(self, factor_num, fully_connect_layer_neural):
        super(AlphaNet_V1, self).__init__()
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
    def __init__(self, factor_num, sequence, fully_connect_layer_neural, attention=False):
        super(AlphaNet_LSTM_V1, self).__init__()
        self.factor_num = factor_num  # 108
        self.sequence = sequence
        self.fc2_neuron = fully_connect_layer_neural  # 32
        self.attention = attention

        # Layer
        self.batch = torch.nn.BatchNorm1d(self.sequence * self.factor_num)
        self.lstm = nn.LSTM(self.factor_num, self.fc2_neuron, 3, batch_first=True, bidirectional=True, dropout=0.2)
        self.lstm2 = nn.LSTM(int(self.fc2_neuron * 2), int(self.fc2_neuron / 2), 3, batch_first=True,
                             bidirectional=True, dropout=0.2)
        self.batch2 = torch.nn.BatchNorm1d(int(self.fc2_neuron * 2))
        self.batch3 = torch.nn.BatchNorm1d(self.fc2_neuron)
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()
        self.LeakyReLU = nn.LeakyReLU()
        self.out = nn.Linear(self.fc2_neuron, 1)

    def forward(self, x):
        x = x.reshape(x.shape[0], -1).float()
        x = self.batch(x)
        x = x.reshape(x.shape[0], self.sequence, self.factor_num)

        x, _ = self.lstm(x)  # x.shape: torch.Size([6182, 10, 128])
        x = self.LeakyReLU(x)
        x = torch.transpose(x, 1, 2)  # x.shape: torch.Size([6182, 128, 10])
        x = self.batch2(x)
        x = torch.transpose(x, 1, 2)

        x, (hn, cn) = self.lstm2(x)  # torch.Size([6182, 10, 64])
        if self.attention == True:
            x, _ = self.attention_net(x, hn)
        else:
            x = x[:, -1]  # torch.Size([6182, 64])
        x = self.batch3(x)  # torch.Size([6182, 64])
        x = self.relu(x)
        x = self.dropout(x)
        y_pred = self.out(x)
        return y_pred

    def attention_net(self, lstm_output, final_state):
        # lstm_output : [batch_size, n_step, n_hidden * num_directions(=2)], F matrix
        # final_state : [num_layers(=1) * num_directions(=2), batch_size, n_hidden]

        hidden = torch.cat((final_state[0], final_state[1]), dim=1).unsqueeze(
            2)  # hidden : [batch_size, n_hidden * num_directions(=2), n_layer(=1)]
        attn_weights = torch.bmm(lstm_output, hidden).squeeze(2)  # [batch_size,sequence]
        attn_weights = torch.nn.functional.softmax(attn_weights, 1)  # [batch_size,sequence]   # torch.Size([512, 20])
        # context: [batch_size, n_hidden * num_directions(=2)]
        output = torch.bmm(lstm_output.transpose(1, 2), attn_weights.unsqueeze(2)).squeeze(
            2)  # [batch_size, n_hidden * num_directions(=2)]
        return output, attn_weights