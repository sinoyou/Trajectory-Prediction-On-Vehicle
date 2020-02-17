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
                             [sample_times, length, 2/5]
        :param step: print step
        :param cat_point: 1 <= cat_point < obs_len, then point where rel_y and rel_y_hat start.
        """
        assert trajectories[0]['x'].ndim == 3

        progress = tqdm(range(len(trajectories)))

        for i, trajectory in enumerate(trajectories):
            fig = plt.figure()
            progress.update(1)
            tag = trajectory['tag']
            all_gaussian_output = trajectory['gaussian_output']  # todo draw heat-map
            all_rel_y_hat = trajectory['rel_y_hat']
            single_x = trajectory['x']
            single_y = trajectory['y']
            single_x_rel = trajectory['rel_x']
            single_y_rel = trajectory['rel_y']

            # calc absolute shift from [0, 0]
            start = torch.unsqueeze(single_x[:, cat_point, :], dim=1)
            all_y_hat = rel_to_abs(all_rel_y_hat, start=start)

            if 'title' in trajectory.keys():
                plt.title(trajectory['title'], fontsize=10)

            # plot observed and ground truth trajectory
            plt.plot(single_x[0, :, 0], single_x[0, :, 1], color='darkblue', label='x')
            y_cat_x = torch.cat((start, single_y), dim=1)
            plt.plot(y_cat_x[0, :, 0], y_cat_x[0, :, 1], color='goldenrod', label='y_gt')
            # plot predicted trajectories(may sample many times)
            sample_times = all_y_hat.shape[0]
            for t in range(sample_times):
                # all paths
                all_y_hat_cat_x = torch.cat((start.repeat(sample_times, 1, 1), all_y_hat), dim=1)
                plt.plot(all_y_hat_cat_x[t, :, 0], all_y_hat_cat_x[t, :, 1], color='deeppink', label='y_hat')

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
    :param start: the last step of observation seq. [1/batch_size, 1, 2]
    :return: trajectory -> Tensor[batch_size, length, 2]
    """
    if start is None:
        start = torch.zeros((rel.shape[0], 1, 2))

    if rel.shape[0] != start.shape[0]:
        start = start.repeat(rel.shape[0], 1, 1)

    trajectory = torch.zeros_like(rel)
    trajectory[:, 0, :] = rel[:, 0, :] + start[:, 0, :]
    for i in range(1, trajectory.shape[1]):
        trajectory[:, i, :] = rel[:, i, :] + trajectory[:, i - 1, :]
    return trajectory
