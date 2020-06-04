import argparse
import errno
import glob
import os
import random

import numpy as np
import skimage.io
from PIL import Image
from sklearn.preprocessing import minmax_scale

from src.utils.measures import tucker_measure


def _parse_arguments():
    """
    parse input arguments, run with -h for help
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--outputdir', type=str, help='directory for the outputs.')
    parser.add_argument('--inputdir', type=str, help="directory for the inputs.")
    parser.add_argument('--samples', type=int, default=2, help='how many samples to mix.')
    parser.add_argument('--times', type=int, default=3, help="how many times the nonlinear mixing should be applied.")
    parser.add_argument('--mix-type', type=str, default="nonlinear", help="type of the mix",
                        choices=["nonlinear", "linear"])
    parser.add_argument('--activation-fun', default="neural_net", type=str,
                        choices=["relu", "tanh", "cos", "anica_mlp", "neural_net"],
                        help="nonlinear function used in the flow mixing model. Used if 'nonlinear' "
                                             "mixing was specified as the mixing type.")
    parser.add_argument('--activation-fun-for-neural-net', type=str, choices=["relu", "tanh", "sigmoid"],
                        default="tanh", help="activation function - if a neural net was chosen as the mixing.")
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--nr-examples', type=int, default=5, help="how many mixes to produce")
    args = parser.parse_args()
    return args


def sigmoid(x):
    return 1. / (1. + np.exp(-x))


def relu(x):
    return np.maximum(x, 0)


def mlp(rs, Z, dim, fun):
    if fun == "sigmoid":
        f = sigmoid
    elif fun == "relu":
        f = relu
    elif fun == "tanh":
        f = np.tanh
    else:
        raise ValueError("Unknown activation function {}".format(fun))
    A = rs.normal(size=(dim, dim))
    b = rs.normal(dim)
    K1 = f(np.matmul(Z, A) + b)
    B = rs.normal(size=(dim, dim))
    b = rs.normal(dim)
    K2 = f(np.matmul(K1, B) + b)
    return K2


def flow_mixing(rs, Z, dim, args):
    for t in range(args.times):
        A = rs.normal(size=(dim, dim))
        hdim = dim // 2
        H = rs.normal(size=(hdim, hdim))

        u, s, vh = np.linalg.svd(A, full_matrices=True)
        if args.mix_type != "nonlinear":
            Y = np.dot(Z, A)
        else:
            Y = np.dot(Z, np.matmul(u, vh))

        i = t % 2
        j = 0 if i == 1 else 1

        X1 = Y[:, i::2]
        X2 = Y[:, j::2]

        if args.activation_fun == "relu":
            Y1 = np.maximum(np.dot(X2, H) + rs.normal(size=hdim), 0) + X1
        elif args.activation_fun == "tanh":
            Y1 = np.tanh(np.dot(X2, H) + rs.normal(size=hdim)) + X1
        elif args.activation_fun == "cos":
            Y1 = np.cos(np.dot(X2, H) + rs.normal(size=hdim)) + X1
        elif args.activation_fun == "anica_mlp":
            Y1 = np.tanh(np.dot(rs.normal(size=(hdim, args.batch_size)), np.tanh(np.dot(X2, H)))) + X1
        elif args.activation_fun == "neural_net":
            Y1 = mlp(rs, X2, dim // 2, args.activation_fun_for_neural_net) + X1
        else:
            raise ValueError("Unknown mixing type {}".format(args.mix_type))

        Y2 = X2
        R = np.zeros(Y.shape)
        R[:, i::2] = Y1
        R[:, j::2] = Y2

        Z = minmax_scale(R)
    return Z


def nonlinear_mixing(args):
    all_pictures = list(glob.glob(args.inputdir))
    tmp = random.sample(all_pictures, args.samples)
    images = np.c_[[skimage.io.imread(_, as_gray=True).flatten() for _ in tmp]].T
    shape = skimage.io.imread(tmp[0], as_gray=True).shape

    batch_size, dim = images.shape
    rs = np.random.RandomState(seed=args.seed)
    Z = images

    tm = 1
    while tm > .9:
        Z = flow_mixing(rs, Z, dim, args.times)
        tm = tucker_measure(Z, images)
        print("Tucker for mix: " + str(tm))

    return Z * 255, shape, images


def plot_and_save_pictures(Z, args, images, it, shape):
    if not os.path.exists(args.outputdir + it + "/"):
        try:
            os.makedirs(args.outputdir + it + "/")
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise
    for i in range(images.shape[1]):
        im = Image.fromarray(images[:, i].reshape(shape))
        im.save(args.outputdir + it + "/" + str(i) + "-data_orig_img_" + args.mix_type + ".png")
    for i in range(Z.shape[1]):
        im = Image.fromarray(np.uint8(Z[:, i].reshape(shape)))
        im.save(args.outputdir + it + "/" + str(i) + "-data_mix_img_" + args.mix_type + ".png")


def linear_mixing(args):
    all_pictures = list(glob.glob(args.inputdir))
    tmp = random.sample(all_pictures, args.samples)
    images = np.c_[[skimage.io.imread(_, as_gray=True).flatten() for _ in tmp]].T
    shape = skimage.io.imread(tmp[0], as_gray=True).shape

    batch_size, dim = images.shape
    rs = np.random.RandomState(seed=args.seed)

    A = np.eye(dim, dim) + 2 * rs.uniform(size=(dim, dim))
    A = A + A.T
    Z = np.abs(np.dot(images, A))
    tm = tucker_measure(Z, images)
    print("Tucker for mix: " + str(tm))

    return minmax_scale(Z) * 255, shape, images


if __name__ == "__main__":
    args = _parse_arguments()
    for i in range(0, args.nr_examples):
        if args.mix_type == "nonlinear":
            Z, shape, images = nonlinear_mixing(args)
        elif args.mix_type == "linear":
            Z, shape, images = linear_mixing(args)
        else:
            raise ValueError("Unknown argument {}".format(args.mix_type))
        plot_and_save_pictures(Z, args, images, str(i), shape)
