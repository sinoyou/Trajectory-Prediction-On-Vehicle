from torch import nn
import torch
import numpy as np


def make_mlp(dim_list, activation='relu', batch_norm=False, dropout=0):
    """Factory for multi-layer perceptron."""
    layers = []
    for dim_in, dim_out in zip(dim_list[:-1], dim_list[1:]):
        layers.append(nn.Linear(dim_in, dim_out))
        if batch_norm:
            layers.append(nn.BatchNorm1d(dim_out))
        if activation == 'relu':
            layers.append(nn.ReLU())
        elif activation == 'leakyrelu':
            layers.append(nn.LeakyReLU())
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))
    return nn.Sequential(*layers)


def l2_loss(pred_traj, pred_traj_gt):
    """
    Input:
    - pred_traj: Tensor of shape (batch, seq_len, 2). Predicted trajectory.
    - pred_traj_gt: Tensor of shape (1/batch, seq_len, 2). Ground truth predictions.
    Output:
    - loss: l2 loss [batch, seq_len, 1]
    """
    loss = (pred_traj_gt - pred_traj) ** 2
    return torch.sqrt(torch.sum(loss, dim=2, keepdim=True))


def neg_likelihood_gaussian_pdf_loss(gaussian_output, target):
    """
    Negative log likelihood loss based on 2D Gaussian Distribution
    :param gaussian_output: Tensor[batch_size, pred_length, 5] [mu_x, mu_y, sigma_x, sigma_y, cor]
    :param target: Tensor[batch_size, pred_length, 2]
    :return: loss -> Tensor[batch_size, pred_length, 1]
    """
    mu_x, mu_y, sigma_x, sigma_y, cor = torch.split(gaussian_output, 1, dim=2)
    tar_x, tar_y = torch.split(target, 1, dim=2)

    # print(mu_x, mu_y)
    # print(tar_x, tar_y)
    # print(sigma_x, sigma_y)

    def gaussian_pdf(x, y, mux, muy, sx, sy, rho):
        """
        2D Gaussian PDF calculation
        ref: https://zh.wikipedia.org/wiki/多元正态分布
        """
        normx, normy = torch.sub(x, mux), torch.sub(y, muy)
        sxsy = torch.mul(sx, sy)
        z1 = torch.div(torch.pow(normx, 2), torch.pow(sx, 2))
        z2 = torch.div(torch.pow(normy, 2), torch.pow(sy, 2))
        z3 = 2 * torch.div(torch.mul(rho, torch.mul(normx, normy)), sxsy)
        z = torch.sub(torch.add(z1, z2), z3)
        index = torch.div(z, torch.mul(torch.sub(torch.pow(rho, 2), 1), 2))
        pdf = torch.exp(index)
        norm_pdf = torch.div(pdf, torch.mul(2 * np.pi, torch.mul(sxsy, torch.sqrt(torch.sub(1, torch.pow(rho, 2))))))
        # norm_pdf = pdf / (2 * np.pi * sxsy * torch.sqrt(1 - rho ** 2))
        return norm_pdf

    step = 1e-2
    pdf1 = gaussian_pdf(tar_x, tar_y, mu_x, mu_y, sigma_x, sigma_y, cor)
    pdf2 = gaussian_pdf(tar_x + step, tar_y, mu_x, mu_y, sigma_x, sigma_y, cor)
    pdf3 = gaussian_pdf(tar_x, tar_y + step, mu_x, mu_y, sigma_x, sigma_y, cor)
    pdf4 = gaussian_pdf(tar_x + step, tar_y + step, mu_x, mu_y, sigma_x, sigma_y, cor)

    pdf_ave = (pdf1 + pdf2 + pdf3 + pdf4) / 4.0
    # print(torch.cat((tar_x, tar_y, pdf1, pdf2, pdf3, pdf4), dim=-1))
    # print(torch.cat((mu_x, mu_y, sigma_x, sigma_y, cor, pdf_ave), dim=-1))
    epsilon = 1e-14
    pdf_ave = torch.clamp(pdf_ave, min=epsilon, max=float('inf'))

    loss = - torch.log(pdf_ave)

    assert loss.shape[0] == gaussian_output.shape[0]
    assert loss.shape[1] == gaussian_output.shape[1]
    assert loss.shape[2] == 1

    return loss


def get_2d_gaussian(model_output):
    """
    Transform model's output into 2D Gaussian format
    :param model_output: Tensor[batch_size, pred_length, 5]
    :return: gaussian_output -> Tensor[batch_size, pred_length, 5]
    """
    mu_x = model_output[..., 0]
    mu_y = model_output[..., 1]
    sigma_x = torch.exp(model_output[..., 2])
    sigma_y = torch.exp(model_output[..., 3])
    cor = torch.tanh(model_output[..., 4])
    return torch.stack((mu_x, mu_y, sigma_x, sigma_y, cor), dim=-1)


def gaussian_sampler(mux, muy, sx, sy, rho):
    """
    Use random sampler to samp;e 2D points from gaussian distribution.
    :return: one 2D point (x, y)
    """
    # Extract mean
    mean = (mux, muy)
    # Extract covariance matrix
    cov = ((sx * sx, rho * sx * sy), (rho * sx * sy, sy * sy))
    # Sample a point from the multiplytivariate normal distribution
    x = np.random.multivariate_normal(mean, cov, size=1)
    return x[0][0], x[0][1]
