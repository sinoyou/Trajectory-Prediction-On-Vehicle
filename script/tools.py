import logging
import sys
import os

import torch
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt


class Recorder:
    """
    Designed specially for recording multiple type logging information.
    # 1. Plot trajectories and distribution on the board.
    """

    def __init__(self, board_dir):
        # log info
        FORMAT = '[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s'
        logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
        self.logger = logging.getLogger(__name__)
        self.writer = SummaryWriter(os.path.join('runs', board_dir))

    def plot_trajectory(self, trajectories: list, step):
        """
        Plot trajectory on the board
        :param trajectories: list of dicts. dict {'tag', 'x', 'y', 'rel_x', 'rel_y', 'gaussian_output'}.
                             [batch_size, length, 2/5]
        :param step: print step
        """
        assert trajectories[0]['x'].ndim == 3

        for batch_trajectory in trajectories:
            tag = batch_trajectory['tag']
            batch_gaussian_output = batch_trajectory['gaussian_output']
            batch_x = batch_trajectory['x']
            batch_y = batch_trajectory['y']
            batch_rel_x = batch_trajectory['rel_x']
            batch_rel_y = batch_trajectory['rel_y']

            # calc absolute shift from [0, 0]
            batch_x_zero = rel_to_abs(batch_rel_x)
            batch_y_zero = rel_to_abs(batch_rel_y)
            batch_y_hat_zero = rel_to_abs(batch_gaussian_output[:, :, 0:2])
            fig = plt.figure()
            for t in range(batch_rel_x.shape[0]):
                # all path starts from [0,0]
                plt.subplot(1, 2, 1)
                plt.title('Trajectory start from [0,0]')
                plt.plot(batch_x_zero[t, :, 0], batch_x_zero[t, :, 1], color='darkblue', label='x')
                plt.plot(batch_y_zero[t, :, 0], batch_y_zero[t:, :, 1], color='goldenrod', label='y_gt')
                plt.plot(batch_y_hat_zero[t, :, 0], batch_y_hat_zero[t, :, 1], color='deeppink', label='y_hat')
                plt.legend(loc=2)
                # original absolute path
                plt.subplot(1, 2, 2)
                plt.title('Original absolute path')
                plt.plot(batch_x[t, :, 0], batch_x[t, :, 1], color='darkblue', label='x')
                plt.plot(batch_y[t, :, 0], batch_y[t, :, 1], color='goldenrod', label='y_gt')
                plt.legend(loc=2)
            self.writer.add_figure(tag=str(tag), figure=fig, global_step=step)


def abs_to_rel(trajectory):
    """
    Transform absolute location into relative location to last step.
    Default: n length trajectory can only get n-1 rel shift, so the first step is [0,0]
    :param trajectory: Tensor[batch_size, length, 2]
    :return: rel_trajectory -> Tensor[batch_size, length, 2]
    """
    rel = torch.zeros_like(trajectory)
    for i in range(0, rel.shape[1]):
        rel[:, i, :] = torch.sub(trajectory[:, i, :], trajectory[:, i - 1, :])
    return rel


def rel_to_abs(rel):
    """
    Transform relative location into abs locatio.
    Default: the last step of the first predicted step is [0,0]
    :param rel: Tensor[batch_size, length, 2]
    :return: trajectory -> Tensor[batch_size, length, 2]
    """
    trajectory = torch.zeros_like(rel)
    trajectory[:, i, :] = rel[:, 0, :]
    for i in range(1, trajectory.shape[1]):
        trajectory[:, i, :] = torch.add(rel[:, i, :], trajectory[:, i - 1, :])
    return trajectory
