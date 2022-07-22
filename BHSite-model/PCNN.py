import torch.nn as nn
import torch


class PCNN(nn.Module):
    def __init__(self, sole_plane, norm_layer=None):
        super(PCNN, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d

        self.conv_s = nn.Conv1d(in_channels=4, out_channels=sole_plane, kernel_size=22, stride=1)
        self.pool_s = nn.MaxPool1d(kernel_size=4, stride=4)

        self.conv_h = nn.Conv1d(in_channels=7, out_channels=sole_plane, kernel_size=1, stride=1)

        self.bn = norm_layer(sole_plane)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, s, h):
        s = self.conv_s(s)
        s = self.bn(s)
        s = self.relu(s)

        s = self.pool_s(s)

        h = self.conv_h(h)
        h = self.bn(h)
        h = self.relu(h)

        cat = torch.cat((s, h), dim=1)

        return cat


def pcnn(sole_plane, **kwargs):
    return PCNN(sole_plane, **kwargs)