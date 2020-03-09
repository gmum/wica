import os
from argparse import ArgumentParser

from src.train.trainer_builder import TrainerBuilder


def get_parser():
    parser = ArgumentParser()
    parser.add_argument('--save_raw', type=bool, default=False, help="save the raw or results in png")
    parser.add_argument('--data-path', default='../data/ica', help="path to the data")
    parser.add_argument('--num-epochs', type=int, default=5, help="number of epochs")
    parser.add_argument('--lr', type=float, default=1e-3, help="the learning rate")
    parser.add_argument('--batch-size', type=int, default=1024, help='size of one batch')
    parser.add_argument('--beta', type=float, default=10, help='independence scaling')
    parser.add_argument('--cuda', action='store_true', help='whether to use cuda')
    parser.add_argument('--rec-loss', choices=["mse", "bce"], default="mse",
                        help='type of the reconstruction error function')
    parser.add_argument('--folder', default="../results", help="output folder")
    parser.add_argument('--save-every', type=int, default=1, help="how often to save the images and model")
    parser.add_argument('--latent-dim', type=int, default=2, help="latent dimension")
    parser.add_argument('--normalize-img', action='store_true', help="whether to apply normlization to input images")
    parser.add_argument('--power', help="power argument in scaling", default=1, type=float)
    parser.add_argument('--number_of_gausses', type=int, help="how many gausses to use for the weighting", default=2)
    return parser


def make_dirs(args):
    os.makedirs(args.folder, exist_ok=True)
    os.makedirs(os.path.join(args.folder, "images"), exist_ok=True)


if __name__ == '__main__':
    args = get_parser().parse_args()
    make_dirs(args)
    trainer = TrainerBuilder.get_trainer(args)
    trainer.train(args)
