import torch.nn as nn
from torch.nn.modules.container import ModuleList


class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Encoder, self).__init__()
        self.latent_dim = latent_dim
        self.input_dim = input_dim

    def forward(self, x):
        return NotImplementedError


class FCEncoder(Encoder):
    """
    Implements a fully connected encoder.
    """

    def __init__(self, input_dim, latent_dim, activations, hidden_dims):
        """
        :param input_dim: input dimension
        :param latent_dim: latent_dimension
        :param activations: array of activations for each layer
        :param hidden_dims: array of hidden_dims
        """

        super(FCEncoder, self).__init__(input_dim, latent_dim)
        self.activations = activations
        self.hidden_dims = hidden_dims

        self.__build_layers()

    def __build_layers(self):
        self.layers = ModuleList([nn.Linear(self.input_dim, self.hidden_dims[0])])
        self.layers.extend(
            [nn.Linear(self.hidden_dims[i], self.hidden_dims[i + 1]) for i in range(0, len(self.hidden_dims) - 1)])
        self.layers.append(nn.Linear(self.hidden_dims[-1], self.latent_dim))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = self.activations[i](layer(x))
        return x
