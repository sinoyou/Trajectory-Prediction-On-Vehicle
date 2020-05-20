import logging
import sys
import torch
import math
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import os

from script.cuda import to_device
from script.visualization import plot_sample_trajectories, plot_gaussian_ellipse, plot_potential_zone

confidence = 5.991  # it's not used for plotting potential zone.
ellipse_args = {'ec': 'blue', 'fill': False, 'lw': 1, 'alpha': 0.5}
plot_args = {'lw': 2, 'alpha': 0.5, 'marker': '*'}
patch_args = {'alpha': 0.9}


class Recorder:
    """
    Designed specially for recording multiple type logging information.
    """

    def __init__(self, summary_path='default', board=True, logfile=True, stream=True):
        """
        :param summary_path: path for saving summary and log file.
        :param board: T/F, if need summary writer.
        :param logfile: T/F, if need to generate log file.
        :param stream: T/F, if need to show
        """
        saved_summary_filepath = '{}/'.format(summary_path)
        if not os.path.exists(saved_summary_filepath):
            os.makedirs(saved_summary_filepath)
        # board
        if board:
            self.writer = SummaryWriter(saved_summary_filepath)
        else:
            self.writer = None
        # log info
        FORMAT = '[%(levelname)s %(asctime)s: %(filename)s: %(lineno)4d]: %(message)s'
        datefmt = '[%Y-%m-%d %H:%M:%S]'
        self.logger = logging.getLogger(name=saved_summary_filepath)
        file_handler = logging.FileHandler(filename=os.path.join(saved_summary_filepath, 'runner.log'))
        stream_handler = logging.StreamHandler(stream=sys.stdout)

        formatter = logging.Formatter(FORMAT, datefmt)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        stream_handler.setLevel(logging.INFO)
        stream_handler.setFormatter(formatter)

        if stream:
            self.logger.addHandler(stream_handler)
        if logfile:
            self.logger.addHandler(file_handler)
        self.logger.setLevel(logging.INFO)

    def plot_trajectory(self, trajectories, step, cat_point, mode, relative):
        """
        Plot trajectory on the board
        :param trajectories: list of dicts. dict {'tag', 'x', 'y', 'rel_x', 'rel_y', 'pred_distribution'}.
                             [sample_times, length, 2/5]
        :param step: print step
        :param cat_point: 1 <= cat_point < obs_len, then point where rel_y and rel_y_hat start.
        :param mode: plot mode. 1 - sample trajectories, 2 - gaussian ellipse, 3 - potential field
        :param relative: if the prediction is on the relative offset.
        """
        progress = tqdm(range(len(trajectories)))

        # count modes
        num_mode = 0
        for i in range(0, int(math.log(mode, 2) + 1)):
            if mode & int((2 ** i)) > 0:
                num_mode += 1

        for i, trajectory in enumerate(trajectories):
            progress.update(1)
            tag = trajectory['tag']
            abs_pred_distribution = trajectory['abs_pred_distribution']
            abs_y_hat = trajectory['abs_y_hat']
            abs_x = trajectory['abs_x']
            abs_y = trajectory['abs_y']

            start = np.expand_dims(abs_x[:, cat_point, :], axis=1)

            fig, subplots = plt.subplots(1, num_mode, figsize=(num_mode * 4, 4), sharex=True, sharey=True)
            if num_mode == 1:
                subplots = [subplots]

            if 'title' in trajectory.keys():
                fig.suptitle(trajectory['title'], fontsize=10)

            subplot_cnt = 0
            # Plot 1: Plot predicted sample trajectories.
            if (mode & 1) != 0:
                plot_sample_trajectories(subplot=subplots[subplot_cnt], abs_x=abs_x, abs_y=abs_y,
                                         start=start, abs_y_hat=abs_y_hat, line_args=plot_args)
                subplot_cnt += 1

            # Plot 2: Plot predicted gaussian Ellipse.
            if (mode & 2) != 0:
                plot_gaussian_ellipse(subplot=subplots[subplot_cnt], abs_x=abs_x, abs_y=abs_y, start=start,
                                      gaussian_output=abs_pred_distribution, confidence=confidence,
                                      ellipse_args=ellipse_args, line_args=plot_args)
                subplot_cnt += 1

            if (mode & 4) != 0:
                plot_potential_zone(subplot=subplots[subplot_cnt], abs_x=abs_x, abs_y=abs_y, start=start,
                                    gaussian_output=abs_pred_distribution,
                                    patch_args=patch_args, line_args=plot_args)

            plt.legend(loc=2)
            self.writer.add_figure(tag=str(tag), figure=fig, global_step=step)

        progress.close()

    def close(self):
        if self.writer:
            self.writer.close()


def abs_to_rel(trajectory):
    """
    Transform absolute location into relative location to last step.
    Default: n length trajectory can only get n-1 rel shift, so the first step is [0,0]
    :param trajectory: Tensor[batch_size, length, 2]
    :return: rel_trajectory -> Tensor[batch_size, length, 2]
    """
    rel = torch.zeros_like(trajectory)
    for i in range(1, rel.shape[1]):
        rel[:, i, :] = torch.sub(trajectory[:, i, :], trajectory[:, i - 1, :])
    return rel


def rel_to_abs(rel, start):
    """
    Transform relative location into abs location.
    :param rel: Tensor[..., length, 2]
    :param start: the last step of observation seq. [..., 1, 2]
    :return: trajectory -> Tensor[..., length, 2]
    """
    if start is None:
        start = torch.zeros((rel.shape[0], 2), device=rel.device)

    trajectory = torch.zeros_like(rel)
    trajectory[..., 0, :] = rel[..., 0, :] + start
    for i in range(1, trajectory.shape[-2]):
        trajectory[..., i, :] = rel[..., i, :] + trajectory[..., i - 1, :]
    return trajectory


def rel_distribution_to_abs_distribution(name, rel_pred, start):
    if name == '2d_gaussian':
        return rel_gaussian_to_abs_gaussian(rel_pred, start)
    elif name == 'mixed':
        return rel_mixed_to_abs_mixed(rel_pred, start)
    else:
        raise Exception('No distribution transformer support for {}'.format(name))


def rel_gaussian_to_abs_gaussian(rel_pred_distribution, start):
    """
    Based on the assumption that relative gaussian distribution predictions are independent each other.
    Transform from relative location prediction to absolute location prediction.
    :param rel_pred_distribution: predicted relative distribution. shape -  [..., pred_len, 5]
    :param start: last location of observation. [batch_size, 2]
    :return:
    """
    abs_pred_distribution = rel_pred_distribution.clone()
    abs_pred_distribution[..., 0, 0:2] = start + rel_pred_distribution[..., 0, 0:2]
    abs_pred_distribution[..., 0, 2:5] = rel_pred_distribution[..., 0, 2:5]

    def transform_to_parameter(mux, muy, sxsx, sysy, sxsy):
        """
        Transformer 2D gaussian matrix form to parameter form
        MATRIX FORM:
        [mux, muy]
            +
        [
            [sxsx, sxsy]
            [sxsy, sysy]
        ]
        """
        sx, sy = torch.sqrt(sxsx), torch.sqrt(sysy)
        rho = sxsy / (sx * sy)
        parameter_cat = torch.cat([mux, muy, sx, sy, rho], dim=-1)
        return parameter_cat

    for step in range(1, rel_pred_distribution.shape[-2]):
        pre_mux, pre_muy, pre_sx, pre_sy, pre_rho = \
            torch.split(abs_pred_distribution[..., step - 1, :], 1, dim=-1)
        cur_mux, cur_muy, cur_sx, cur_sy, cur_rho = \
            torch.split(rel_pred_distribution[..., step, :], 1, dim=-1)
        sum_mux = pre_mux + cur_mux
        sum_muy = pre_muy + cur_muy
        sum_sxsx = pre_sx ** 2 + cur_sx ** 2
        sum_sysy = pre_sy ** 2 + cur_sy ** 2
        sum_sxsy = pre_sx * pre_sy * pre_rho + cur_sx * cur_sy * cur_rho
        abs_pred_distribution[..., step, :] = \
            transform_to_parameter(sum_mux, sum_muy, sum_sxsx, sum_sysy, sum_sxsy)

    return abs_pred_distribution


def rel_mixed_to_abs_mixed(rel_pred_distribution, start):
    """
    Transform from rel to abs distribution. Notice that distribution along x and y is seperate.
    :param rel_pred_distribution: [..., pred_len, 5]
    :param start: [batch_size, 2]
    :return: [..., pred_len, 5]. Notice that mixed distribution only use 4 parameters, so the last one is useless.
    """
    abs_pred_distribution = rel_pred_distribution.clone()
    abs_pred_distribution[..., 0, 0:2] = start + rel_pred_distribution[..., 0, 0:2]
    abs_pred_distribution[..., 0, 2:4] = rel_pred_distribution[..., 0, 2:4]
    for step in range(1, rel_pred_distribution.shape[-2]):
        abs_pred_distribution[..., step, 0:4] = rel_pred_distribution[..., step, 0:4] + \
                                                abs_pred_distribution[..., step - 1, 0:4]
    return abs_pred_distribution
