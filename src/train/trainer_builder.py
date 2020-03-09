from torch.utils.data import DataLoader

from src.data_handling.datasets import FlattenedPicturesDataset
from src.decoders.decoder_provider import DecoderProvider
from src.encoders.encoder_provider import EncoderProvider
from src.models.loss_functions import *
from src.models.models import *
from src.train.trainers import *


class TrainerBuilder:
    @staticmethod
    def get_trainer(args):
        loss = TrainerBuilder.__get_loss(args)
        dataloaders = TrainerBuilder.__get_dataloaders(args)
        trainer = TrainerBuilder.__get_trainer(args, loss, dataloaders)
        return trainer

    @staticmethod
    def __get_trainer(args, loss, dataloaders):
        encoder = EncoderProvider.get_encoder(args.latent_dim)
        decoder = DecoderProvider.get_decoder(args.latent_dim, args.normalize_img)

        model = Autoencoder(encoder, decoder)
        return IndependenceAETrainer(model, loss, dataloaders, args.cuda)

    @staticmethod
    def __get_loss(args):
        reg_loss = ReconstructionLoss.get_rec_loss(args.rec_loss)
        ind_loss = WeightedICALossFunction(args.power, args.number_of_gausses, cuda=args.cuda)
        return JoinedLoss(ind_loss, reg_loss, args.beta)

    @staticmethod
    def __get_dataloaders(args):
        train_dataset = FlattenedPicturesDataset(args.data_path, 2, 2)
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

        test_dataset = FlattenedPicturesDataset(args.data_path, 2, 2)
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size)
        return train_dataloader, test_dataloader
