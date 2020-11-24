import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        emb_dim = 300
        hidden_size = 100

        self.emb = nn.Embedding(1000, 100, padding_idx=0)

        self.projection = nn.Linear(hidden_size, hidden_size)
        self.rnn = nn.LSTM(hidden_size, hidden_size,
            num_layers=2, bidirectional = True, batch_first = True,
            dropout = 0.0)

        self.lin_1 = nn.Linear(hidden_size * 2 * 2, hidden_size)
        self.lin_2 = nn.Linear(hidden_size, hidden_size)
        self.lin_3 = nn.Linear(3, hidden_size)

    def forward(self, x_hyp, x_pre):

        pre_embed = self.emb(x_pre)
        hyp_embed = self.emb(x_hyp)

        _, (pre_ht, _) = self.rnn(pre_embed)
        _, (hyp_ht, _) = self.rnn(hyp_embed)

        x_out = F.relu(self.lin_1(torch.cat((pre_ht, hyp_ht), dim=1).squeeze()))
        x_out = F.relu(self.lin_2(x_out))

        return F.log_softmax(self.lin_3(x_out), dim=1)
