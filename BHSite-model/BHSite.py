from PCNN import pcnn
from SEResNet import se_resnet
from GRUNet import gru
from Highway import highway

import torch.nn as nn
import torch


class BHSite(nn.Module):

    def __init__(self, sole_plane, res_block, res_layers,
                 gru_hidden_num, gru_layers, highway_layers):
        super(BHSite, self).__init__()

        """ compute input size of GRUNet """
        gru_input_num = (sole_plane * 2) * ((len(res_layers) - 1) * 2)

        """ compute input size of highway followed by Bi-GRU"""
        highway_input = (gru_hidden_num * 2) * (20 // ((len(res_layers) - 1) * 2))

        self.PCNN = pcnn(sole_plane=sole_plane)
        self.SEResNet = se_resnet(block=res_block, layers=res_layers)
        self.GURNet = gru(input_num=gru_input_num, hidden_num=gru_hidden_num, num_layer=gru_layers)

        self.flatten = nn.Flatten(start_dim=1)

        self.Highway = highway(in_size=highway_input, num_layer=highway_layers)

    def forward(self, s, h):
        cat = self.PCNN(s, h)

        out = self.SEResNet(cat)

        out = torch.transpose(out, 1, 2)
        out = self.GURNet(out)

        out = self.flatten(out)
        out = self.Highway(out)

        return out