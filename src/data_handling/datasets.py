import glob

import numpy as np
import torch
from skimage import io
from torch.utils.data import Dataset


class FlattenedPicturesDataset(Dataset):
    """
    The dataset containing a flattened mix of pictures.
    """

    def __init__(self, path_to_mixes, mix_dim, orig_dim):
        self.path_to_mixes = path_to_mixes
        self.mix_dim = mix_dim
        self.orig_dim = orig_dim
        self.mix = self.flatten_images(self.__get__("mix", self.mix_dim))
        self.orig = self.flatten_images(self.__get__("orig", self.orig_dim))

    def __getitem__(self, index):
        mix = self.mix[index]
        orig = self.orig[index]
        mix = mix / 255.
        orig = orig / 255.
        return mix, orig

    def __get__(self, type_, dim):
        files = []
        for i in range(dim):
            files.append(None)

        for i in range(self.mix_dim):
            files[i] = sorted(glob.glob(self.path_to_mixes + "/*/" + str(i) + "*" + type_ + "*"))

        return files

    @staticmethod
    def flatten_images(signals):
        f = []
        for i in range(len(signals)):
            f.append(None)
            f[i] = np.concatenate([io.imread(_, as_gray=True).flatten() for _ in signals[i]])
        return torch.transpose(torch.Tensor(f), 0, 1)

    def __len__(self):
        return len(self.mix)
