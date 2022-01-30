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
    def __init__(self, factor_num, sequence, fully_connect_layer_neural, attention=False, transformer=False):
        super(AlphaNet_LSTM_V1, self).__init__()
        self.factor_num = factor_num  # 108
        self.sequence = sequence
        self.fc2_neuron = fully_connect_layer_neural  # 32
        self.attention = attention
        self.transformer = transformer
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
        if self.transformer == True:
            self.TransformerLayer = nn.TransformerEncoderLayer(d_model=self.fc2_neuron * 2, nhead=2, batch_first=True)

    def forward(self, x):
        x = x.reshape(x.shape[0], -1).float()
        x = self.batch(x)
        x = x.reshape(x.shape[0], self.sequence, self.factor_num)

        x, _ = self.lstm(x)  # x.shape: torch.Size([6182, 10, 128])
        if self.transformer == True:
            x = self.TransformerLayer(x)  # attn_output = torch.Size([512, 20, 128])
        else:
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

class Res_LSTM(nn.Module):
    def __init__(self, dimention, factor_num, sequence, fully_connect_layer_neural, layer_num=2, transformer=False):
        super(Res_LSTM, self).__init__()
        self.factor_num = factor_num  # 108
        self.sequence = sequence  #
        self.dimention = dimention  #
        self.fc2_neuron = fully_connect_layer_neural  # 32
        self.transformer = transformer

        # Layer
        self.bn1 = torch.nn.BatchNorm1d(self.dimention * self.factor_num * self.sequence)
        self.bn2 = torch.nn.BatchNorm1d(self.fc2_neuron * 2 * self.sequence)
        self.bn3 = torch.nn.BatchNorm1d(self.fc2_neuron * 2)
        if self.transformer == True:
            self.q_metrix = nn.Linear(self.factor_num, self.factor_num)
            self.k_metrix = nn.Linear(self.factor_num, self.factor_num)
            self.v_metrix = nn.Linear(self.factor_num, self.factor_num)
            self.MultiheadAttention = nn.MultiheadAttention(self.factor_num, layer_num, batch_first=True)

        self.lstm = nn.LSTM(self.factor_num, self.fc2_neuron, layer_num, batch_first=True, bidirectional=True,
                            dropout=0.2)
        self.dropout = nn.Dropout(0.2)
        self.LeakyReLU = nn.LeakyReLU()
        self.out = nn.Linear(self.fc2_neuron * 2, 1)

    def forward(self, x):
        x = self.norm_1(x)
        x = torch.transpose(x, 0, 1)  # x.shape: torch.Size([3, 512, 20, 108])

        final, (hn, cn) = self.lstm_layer(x[0], 2)  # torch.Size([512, 20, 128])
        for i in range(1, x.shape[0]):
            add, _ = self.lstm_layer(x[i], 2)
            final = self.skip_connection(final, add)
        # start = torch.Size([512, 20, 128])

        x, _ = self.attention_net(final, hn)
        x = self.bn3(x)
        x = self.LeakyReLU(x)
        x = self.dropout(x)
        y_pred = self.out(x)
        return y_pred

    def norm_1(self, x):
        batch_num, original_shape = x.shape[0], x.shape
        x = x.reshape(batch_num, -1)
        x = self.bn1(x)
        x = x.reshape(original_shape)
        return x

    def norm_2(self, x):
        batch_num, original_shape = x.shape[0], x.shape
        x = x.reshape(batch_num, -1)
        x = self.bn2(x)
        x = x.reshape(original_shape)
        return x

    def lstm_layer(self, x, layer_num):
        if self.transformer == True:
            q = self.q_metrix(x)
            k = self.k_metrix(x)
            v = self.v_metrix(x)
            x, x_weight = self.MultiheadAttention(q, k, v)  # attn_output = torch.Size([512, 20, 128])
        else:
            pass
        # out = torch.Size([512, 20, 128])
        out, (hn, cn) = self.lstm(x)
        return out, (hn, cn)

    def skip_connection(self, origin, add):
        return self.norm_2(origin + add)

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