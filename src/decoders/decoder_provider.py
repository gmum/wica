from src.decoders.decoder import *


class FlattenImagesICADecoder(object):
    def __init__(self):
        pass

    @staticmethod
    def get_decoder(latent_dim, last_activation):
        decoder = FCDecoder(
            latent_dim=latent_dim,
            activations=[nn.ReLU(), last_activation],
            hidden_dims=[128],
            output_dim=2
        )
        return decoder


class DecoderProvider:
    """
    Provides the decoder architecture.
    """
    def __init__(self):
        pass

    @staticmethod
    def get_decoder(latent_dim, normalize_img):
        if normalize_img:
            last_activation = nn.Tanh()
        else:
            last_activation = nn.Sigmoid()
        return FlattenImagesICADecoder.get_decoder(latent_dim, last_activation)
