import torch.nn as nn
from torch.nn.modules.container import ModuleList


class Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super(Decoder, self).__init__()
        self.latent_dim = latent_dim
        self.output_dim = output_dim

    def forward(self, x):
        return NotImplementedError


class FCDecoder(Decoder):
    """
    Implements a fully connected decoder.
    """

    def __init__(self, latent_dim, activations, hidden_dims, output_dim):
        """
        :param latent_dim: dimension of the latent space
        :param output_dim: dimension of the output
        :param activations: array of activations names for each layer
        :param hidden_dims: array of hidden_dims (if different for each layer)
        """
        super(FCDecoder, self).__init__(latent_dim, output_dim)
        self.activations = activations
        self.hidden_dims = hidden_dims

        # build the network
        self.__build_layers()

    def __build_layers(self):
        self.layers = ModuleList([nn.Linear(self.latent_dim, self.hidden_dims[0])])
        self.layers.extend(
            [nn.Linear(self.hidden_dims[i], self.hidden_dims[i + 1]) for i in range(0, len(self.hidden_dims) - 1)])
        self.layers.append(nn.Linear(self.hidden_dims[-1], self.output_dim))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = self.activations[i](layer(x))
        return x
