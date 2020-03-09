from src.encoders.encoder import *


class FlattenImagesICAEncoder:
    def __init__(self):
        pass

    @staticmethod
    def get_encoder(latent_dim):
        encoder = FCEncoder(
            input_dim=2,
            latent_dim=latent_dim,
            activations=[
                nn.Sequential(nn.ReLU()),
                nn.Sequential(nn.Identity())
            ],
            hidden_dims=[128]
        )
        return encoder


class EncoderProvider:
    """
    Provides the encoder architecture.
    """
    def __init__(self):
        pass

    @staticmethod
    def get_encoder(latent_dim):
        return FlattenImagesICAEncoder.get_encoder(latent_dim)
