import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, input_size):
        super(Model, self).__init__()
        self.bnorm = nn.BatchNorm1d(input_size)
        self.lin_a = nn.Linear(input_size, input_size)

        emb_dim = 50
        self.emb = nn.Embedding(8000, emb_dim, padding_idx=0)
        self.rnn = nn.GRU(emb_dim, input_size, batch_first=True)

        self.drop = nn.Dropout(p=0.2)
        self.lin_o = nn.Linear(input_size * 2, 2)

    def forward(self, x, xt):
        x = self.bnorm(x)
        if type(xt) == nn.utils.rnn.PackedSequence:
            xt = nn.utils.rnn.PackedSequence(
                data=self.emb(xt.data), batch_sizes=xt.batch_sizes,
                sorted_indices=None, unsorted_indices=None)
        elif torch.is_tensor(xt):
            xt = self.emb(xt)
        else:
            print("Incorrect Input Type")
        h = self.rnn(xt)[1].squeeze(0)

        x = F.relu(torch.cat((self.lin_a(x), h), dim=1))
        x = self.drop(x)

        return F.log_softmax(self.lin_o(x), dim=1)
