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

    def __init__(self, name='default', filename=None):
        # log info
        FORMAT = '[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s'
        if filename:
            logging.basicConfig(level=logging.INFO, format=FORMAT, filename=filename)
        else:
            logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
        self.logger = logging.getLogger(__name__)
        saved_summary_filepath = '{}/'.format(name)
        if not os.path.exists(saved_summary_filepath):
            os.makedirs(saved_summary_filepath)
        self.writer = SummaryWriter(saved_summary_filepath)

    def plot_trajectory(self, trajectories, step, cat_point, mode, relavtive):
        """
        Plot trajectory on the board
        :param trajectories: list of dicts. dict {'tag', 'x', 'y', 'rel_x', 'rel_y', 'pred_distribution'}.
                             [sample_times, length, 2/5]
        :param step: print step
        :param cat_point: 1 <= cat_point < obs_len, then point where rel_y and rel_y_hat start.
        :param mode: plot mode. 1 - sample trajectories, 2 - gaussian ellipse, 3 - potential field
        :param relavtive: if the prediction is on the relative offset.
        """
        # assert trajectories[0]['x'].ndim == 3  # for pytorch1.4.0
        assert len(trajectories[0]['x'].shape) == 3

        progress = tqdm(range(len(trajectories)))

        # count modes
        num_mode = 0
        for i in range(0, int(math.log(mode, 2) + 1)):
            if mode & int((2 ** i)) > 0:
                num_mode += 1

        for i, trajectory in enumerate(trajectories):
            progress.update(1)
            tag = trajectory['tag']
            pred_distribution = trajectory['pred_distribution']
            abs_y_hat = trajectory['abs_y_hat']
            abs_x = trajectory['abs_x']
            abs_y = trajectory['abs_y']

            # when using relative prediction, gaussian mux and muy should be replaced with absolute for visualization
            if relavtive:
                if i == 0:
                    self.logger.warning(
                        'Notice relative data is using. If distribution is not 2d_gaussian, plot may be unexpected.')
                abs_pred_distribution = np.zeros_like(pred_distribution)
                abs_pred_distribution[:, 0, 0:2] = abs_y_hat[:, 0, 0:2]
                abs_pred_distribution[:, 0, 2:5] = pred_distribution[:, 0, 2:5]

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
                    sx, sy = np.sqrt(sxsx), np.sqrt(sysy)
                    rho = sxsy / (sx * sy)
                    parameter_cat = np.concatenate([mux, muy, sx, sy, rho], axis=-1)
                    assert len(parameter_cat.shape) == 2
                    return parameter_cat

                for step in range(1, pred_distribution.shape[1]):
                    pre_mux, pre_muy, pre_sx, pre_sy, pre_rho = \
                        np.split(abs_pred_distribution[:, step - 1, :], indices_or_sections=5, axis=-1)
                    cur_mux, cur_muy, cur_sx, cur_sy, cur_rho = \
                        np.split(pred_distribution[:, step, :], indices_or_sections=5, axis=-1)
                    sum_mux = pre_mux + cur_mux
                    sum_muy = pre_muy + cur_muy
                    sum_sxsx = pre_sx ** 2 + cur_mux ** 2
                    sum_sysy = pre_sy ** 2 + cur_muy ** 2
                    sum_sxsy = pre_sx * pre_sy * pre_rho + cur_sx * cur_sy * cur_rho
                    abs_pred_distribution[:, step, :] = \
                        transform_to_parameter(sum_mux, sum_muy, sum_sxsx, sum_sysy, sum_sxsy)

                pred_distribution = abs_pred_distribution

            start = np.expand_dims(abs_x[:, cat_point, :], axis=1)

            fig, subplots = plt.subplots(1, num_mode)
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
                                      gaussian_output=pred_distribution, confidence=confidence,
                                      ellipse_args=ellipse_args, line_args=plot_args)
                subplot_cnt += 1

            if (mode & 4) != 0:
                plot_potential_zone(subplot=subplots[subplot_cnt], abs_x=abs_x, abs_y=abs_y, start=start,
                                    gaussian_output=pred_distribution,
                                    patch_args=patch_args, line_args=plot_args)

            plt.legend(loc=2)
            self.writer.add_figure(tag=str(tag), figure=fig, global_step=step)

        progress.close()

    def close(self):
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
    :param rel: Tensor[batch_size, length, 2]
    :param start: the last step of observation seq. [1/batch_size, 1, 2]
    :return: trajectory -> Tensor[batch_size, length, 2]
    """
    if start is None:
        start = to_device(torch.zeros((rel.shape[0], 1, 2)), device=rel.device)

    if rel.shape[0] != start.shape[0]:
        start = start.repeat(rel.shape[0], 1, 1)

    trajectory = torch.zeros_like(rel)
    trajectory[:, 0, :] = rel[:, 0, :] + start[:, 0, :]
    for i in range(1, trajectory.shape[1]):
        trajectory[:, i, :] = rel[:, i, :] + trajectory[:, i - 1, :]
    return trajectory