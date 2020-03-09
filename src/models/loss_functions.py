import torch
import torch.nn as nn
from torch import ones, multinomial
from torch.distributions import MultivariateNormal


class LossFunction:
    def __init__(self):
        pass

    def loss(self, **kwargs):
        raise NotImplementedError


class JoinedLoss(LossFunction):
    """
    Implements joined loss function (reconstruction loss and independence loss).
    """
    def __init__(self, ind_loss, reg_loss, beta, return_separate=False):
        super(JoinedLoss, self).__init__()
        self.beta = beta
        self.ind_loss = ind_loss
        self.reg_loss = reg_loss
        self.return_separate = return_separate
        self.reduction_type = "mean"

    def loss(self, recon_x, x, **ind_loss_kwargs):
        rec_loss = self.reg_loss.loss(recon_x, x)
        if self.return_separate:
            main_loss = rec_loss + self.beta * self.ind_loss.loss(**ind_loss_kwargs)
            return main_loss, rec_loss
        else:
            return rec_loss + self.beta * self.ind_loss.loss(**ind_loss_kwargs)


class ReconstructionLoss(LossFunction):
    """
    Represents the reconstruction loss.
    """
    def __init__(self, ):
        super(ReconstructionLoss, self).__init__()

    def loss(self, recon_x, x):
        raise NotImplementedError

    @staticmethod
    def get_rec_loss(rec_loss):
        if rec_loss == 'mse':
            return MSELossFunction()
        elif rec_loss == 'bce':
            return BinaryCrossEntropyFunction()
        else:
            raise ValueError("Unknown reconstruction loss {}".format(rec_loss))


class MSELossFunction(ReconstructionLoss):
    """
    The Mean Square Error loss.
    """
    def __init__(self):
        super(MSELossFunction, self).__init__()
        self.mse = nn.MSELoss()
        self.reduction_type = "mean"

    def loss(self, recon_x, x):
        return self.mse(recon_x, x)


class BinaryCrossEntropyFunction(ReconstructionLoss):
    """
    The Binary Cross Entropy loss.
    """
    def __init__(self):
        super(BinaryCrossEntropyFunction, self).__init__()
        self.bce = nn.BCELoss()
        self.reduction_type = "mean"

    def loss(self, recon_x, x):
        return self.bce(recon_x, x)


class WeightedICALossFunction(LossFunction):
    """
    The weighted correlation loss (the independence loss).
    """
    def __init__(self, power, number_of_gausses, cuda, z_dim=None):
        super(WeightedICALossFunction, self).__init__()
        self.power = power
        self.number_of_gausses = number_of_gausses
        self.z_dim = z_dim
        self.cuda = cuda
        self.reduction_type = "mean"

    def random_choice_full(self, input, n_samples):
        if n_samples * self.number_of_gausses < input.shape[0]:
            replacement = False
        else:
            replacement = True
        idx = multinomial(ones(input.shape[0]), n_samples * self.number_of_gausses, replacement=replacement)
        sampled = input[idx].reshape(self.number_of_gausses, n_samples, -1)
        return torch.mean(sampled, axis=1)

    def loss(self, z, latent_normalization=True):
        if latent_normalization:
            x = (z - z.mean(axis=0)) / z.std(axis=0)
        else:
            x = z
        dim = self.z_dim if self.z_dim is not None else x.shape[1]
        scale = (1 / dim) ** self.power
        sampled_points = self.random_choice_full(x, dim)

        cov_mat = (scale * torch.eye(dim)).repeat(self.number_of_gausses, 1, 1)
        if self.cuda:
            cov_mat = cov_mat.cuda()

        mvn = MultivariateNormal(loc=sampled_points,
                                 covariance_matrix=cov_mat)

        weight_vector = torch.exp(mvn.log_prob(x.reshape(-1, 1, dim)))
        sum_of_weights = torch.sum(weight_vector, axis=0)

        weight_sum = torch.sum(x * weight_vector.T.reshape(self.number_of_gausses, -1, 1), axis=1)
        weight_mean = weight_sum / sum_of_weights.reshape(-1, 1)

        xm = x - weight_mean.reshape(self.number_of_gausses, 1, -1)
        wxm = xm * weight_vector.T.reshape(self.number_of_gausses, -1, 1)

        wcov = (wxm.permute(0, 2, 1).matmul(xm)) / sum_of_weights.reshape(-1, 1, 1)

        diag = torch.diagonal(wcov ** 2, dim1=1, dim2=2)
        diag_pow_plus = diag.reshape(diag.shape[0], diag.shape[1], -1) + diag.reshape(diag.shape[0], -1, diag.shape[1])

        tmp = (2 * wcov ** 2 / diag_pow_plus)
        triu = torch.triu(tmp, diagonal=1)
        normalize = 2.0 / (dim * (dim - 1))
        cost = torch.sum(normalize * triu) / self.number_of_gausses
        return cost
