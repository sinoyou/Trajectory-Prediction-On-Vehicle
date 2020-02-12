import logging
import sys
import torch
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
from tqdm import tqdm


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

    def plot_trajectory(self, trajectories: list, step, cat_point):
        """
        Plot trajectory on the board
        :param trajectories: list of dicts. dict {'tag', 'x', 'y', 'rel_x', 'rel_y', 'gaussian_output'}.
                             [batch_size, length, 2/5]
        :param step: print step
        :param cat_point: 1 <= cat_point < obs_len, then point where rel_y and rel_y_hat start.
        """
        assert trajectories[0]['x'].ndim == 3

        progress = tqdm(range(len(trajectories)))

        for i, batch_trajectory in enumerate(trajectories):
            fig = plt.figure()
            progress.update(1)
            tag = batch_trajectory['tag']
            batch_gaussian_output = batch_trajectory['gaussian_output']  # todo draw heat-map
            batch_rel_y_hat = batch_trajectory['rel_y_hat']
            batch_x = batch_trajectory['x']
            batch_y = batch_trajectory['y']
            batch_rel_x = batch_trajectory['rel_x']
            batch_rel_y = batch_trajectory['rel_y']

            # calc absolute shift from [0, 0]
            start = torch.unsqueeze(batch_x[:, cat_point, :], dim=1)
            batch_y_hat = rel_to_abs(batch_rel_y_hat, start=start)

            if 'title' in batch_trajectory.keys():
                plt.title(batch_trajectory['title'], fontsize=10)

            for t in range(batch_y.shape[0]):
                # all paths

                plt.plot(batch_x[t, :, 0], batch_x[t, :, 1], color='darkblue', label='x')

                plot_path = torch.cat((start, batch_y), dim=1)
                plt.plot(plot_path[t, :, 0], plot_path[t, :, 1], color='goldenrod', label='y_gt')

                plot_path = torch.cat((start, batch_y_hat), dim=1)
                plt.plot(plot_path[t, :, 0], plot_path[t, :, 1], color='deeppink', label='y_hat')

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
    for i in range(1, rel.shape[1]):
        rel[:, i, :] = torch.sub(trajectory[:, i, :], trajectory[:, i - 1, :])
    return rel


def rel_to_abs(rel, start):
    """
    Transform relative location into abs location.
    :param rel: Tensor[batch_size, length, 2]
    :param start: the last step of observation seq. [batch_size, 1, 2]
    :return: trajectory -> Tensor[batch_size, length, 2]
    """
    if start is None:
        start = torch.zeros((rel.shape[0], 1, 2))

    trajectory = torch.zeros_like(rel)
    trajectory[:, 0, :] = rel[:, 0, :] + start
    for i in range(1, trajectory.shape[1]):
        trajectory[:, i, :] = rel[:, i, :] + trajectory[:, i - 1, :]
    return trajectory
