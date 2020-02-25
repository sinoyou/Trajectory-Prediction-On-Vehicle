import logging
import sys
import torch
import math
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
from tqdm import tqdm

from script.cuda import to_device


class Recorder:
    """
    Designed specially for recording multiple type logging information.
    # 1. Plot trajectories and distribution on the board.
    """

    def __init__(self):
        # log info
        FORMAT = '[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s'
        logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
        self.logger = logging.getLogger(__name__)
        self.writer = SummaryWriter('../runs/')

    def plot_trajectory(self, trajectories, step, cat_point, mode):
        """
        Plot trajectory on the board
        :param trajectories: list of dicts. dict {'tag', 'x', 'y', 'rel_x', 'rel_y', 'gaussian_output'}.
                             [sample_times, length, 2/5]
        :param step: print step
        :param cat_point: 1 <= cat_point < obs_len, then point where rel_y and rel_y_hat start.
        """
        # assert trajectories[0]['x'].ndim == 3  # for pytorch1.4.0
        assert len(trajectories[0]['x'].shape) == 3

        progress = tqdm(range(len(trajectories)))

        # count modes
        num_mode = 0
        for i in range(1, int(math.log(mode, 2) + 1)):
            if mode & i == 0:
                num_mode += num_mode

        for i, trajectory in enumerate(trajectories):
            fig = plt.figure()
            progress.update(1)
            tag = trajectory['tag']
            all_gaussian_output = trajectory['gaussian_output']
            abs_y_hat = trajectory['abs_y_hat']
            abs_x = trajectory['abs_x']
            abs_y = trajectory['abs_y']

            start = torch.unsqueeze(abs_x[:, cat_point, :], dim=1)

            fig, subplots = plt.subplots(num_mode, 1)

            if 'title' in trajectory.keys():
                for subplot in subplots:
                    subplot.title(trajectory['title'], fontsize=10)

            # Plot 1: Plot predicted sample trajectories.
            if mode & 1 != 0:
                pass

            # Plot 2: Plot predicted gaussian Ellipse.
            if mode & 2 != 0:
                pass

            # todo Plot 3: Plot predicted potenfial zone according to gaussian Ellipse.
            if mode & 4 != 0:
                raise Exception('Visualization Mode 3 not implemented.')

            plt.legend(loc=2)
            self.writer.add_figure(tag=str(tag), figure=fig, global_step=step)

        progress.close()


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
