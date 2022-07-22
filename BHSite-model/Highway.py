import torch.nn as nn


def perceptron(in_size, out_size):
    return nn.Linear(in_size, out_size)


class Highway(nn.Module):
    def __init__(self, in_size, num_layers, reduction=4):
        super(Highway, self).__init__()

        self.in_size = in_size
        self.num_layers = num_layers
        self.hidden_size = int(in_size // reduction)

        self.nonlinear = self._make_layer()
        self.linear = self._make_layer()
        self.gate = self._make_layer()
        self.discriminator = nn.Linear(self.hidden_size, 1)

        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def _make_layer(self):
        layers = nn.ModuleList()

        layers.append(perceptron(self.in_size, self.hidden_size))

        for _ in range(1, self.num_layers):
            layers.append(perceptron(self.hidden_size, self.hidden_size))

        return layers

    def forward(self, x):

        for i in range(self.num_layers):
            gate = self.sigmoid(self.gate[i](x))

            nonlinear = self.relu(self.nonlinear[i](x))

            linear = self.linear[i](x)

            x = gate * nonlinear + (1 - gate) * linear

        out = self.sigmoid(self.discriminator(x))

        return out


def highway(in_size, num_layer, **kwargs):
    return Highway(in_size, num_layer, **kwargs)