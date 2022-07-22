import torch.nn as nn


class GRU(nn.Module):

    def __init__(self, input_num, hidden_num, num_layer,
                 bidirectional=True, dropout=0.2):
        super(GRU, self).__init__()

        self.GRU_layer = nn.GRU(input_size=input_num, hidden_size=hidden_num,
                                num_layers=num_layer, batch_first=True,
                                bidirectional=bidirectional, dropout=dropout)
        self.hidden = None

    def forward(self, x):
        out, self.hidden = self.GRU_layer(x)
        return out


def gru(input_num, hidden_num, num_layer, **kwargs):
    return GRU(input_num, hidden_num, num_layer, **kwargs)