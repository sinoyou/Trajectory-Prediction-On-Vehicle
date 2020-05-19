import numpy as np
import random
import pandas as pd
import torch

from script.tools import rel_to_abs, abs_to_rel


class SingleKittiDataLoader:
    """
    Kitti DataLoader for single object's trajectory sequence. 
    Suitable for vanilla, seq2seq and etc which not considering social interaction in a scene. 
    """

    def __init__(self, file_path, batch_size, trajectory_length, device, mode, train_leave, recorder, seed=17373321,
                 valid_scene=None, batch_fragment=False):
        self.count = 0
        self.batch_size = batch_size
        self.seq_len = trajectory_length
        self.batch_ptr = 0
        self.device = device
        self.mode = str.lower(mode)
        self.recorder = recorder
        self.batch_fragment = batch_fragment

        # read raw data. 
        raw_data = pd.read_csv(file_path)

        # args check
        assert not (self.mode == 'valid' and valid_scene is None)
        assert self.mode in ['train', 'valid']

        # get train data
        if train_leave is not None:
            if isinstance(train_leave, int):
                train_leave = [train_leave]
            self.recorder.logger.info('Scenes {} are left not for training.'.format(train_leave))
            leaves = [raw_data['scene'] == s for s in train_leave]
            mask = leaves[0]
            for i in range(1, len(leaves)):
                mask = mask | leaves[i]
            self.train_data = raw_data[~mask]
        else:
            self.train_data = raw_data

        # get valid data
        if valid_scene is not None:
            if isinstance(valid_scene, int):
                valid_scene = [valid_scene]
            # valid scene
            self.recorder.logger.info('Scenes {} are used for validation.'.format(valid_scene))
            targets = [raw_data['scene'] == s for s in valid_scene]
            mask = targets[0]
            for i in range(1, len(targets)):
                mask = mask | targets[i]
            self.valid_data = raw_data[mask]
        else:
            self.valid_data = None

        # get mean and std from training data.
        self.norm_targets = ['loc_x', 'loc_y', 'loc_z']
        self.norm_metric = dict()
        self.get_mean_std()

        if self.mode == 'train':
            self.data = self.preprocess(self.train_data)
        else:
            self.data = self.preprocess(self.valid_data)

        self.count = len(self.data)
        random.seed(seed)
        random.shuffle(self.data, random=random.random)

        # print summary. 
        self.recorder.logger.info('Count = {}, Batch Size = {}, Iteration = {}, Device = {}'.format(
            self.count, self.batch_size, self.__len__(), self.device
        ))
        if self.__len__() <= 0:
            raise Exception('No enough data for a batch size = {}'.format(self.batch_size))

    def preprocess(self, filter_raw):
        """
        process loaded data into list with length = count
        each unit in a list -> [trajectory_length, 2]
        """
        data = list()

        # norm process
        def norm(row):
            for target in self.norm_targets:
                row[target] = (row[target] - self.norm_metric[target + '_mean']) / self.norm_metric[target + '_std']
            return row

        filter_raw = filter_raw.apply(norm, axis=1)

        # take out single object sequence and slice it into seq_len.
        scenes = filter_raw['scene'].unique()
        for scene in scenes:
            raw_scene = filter_raw[filter_raw['scene'] == scene]
            raw_scene = raw_scene.sort_values(by='frame')
            vru_ids = raw_scene['id'].unique()
            for vru in vru_ids:
                vru_traj = raw_scene[raw_scene['id'] == vru][['loc_x', 'loc_z']]
                # window(=seq_len) shift on slice
                vru_traj_np = np.array(vru_traj)
                for start in range(0, vru_traj_np.shape[0] - self.seq_len + 1):
                    unit = np.zeros((self.seq_len, 2))
                    unit[:, :] = vru_traj_np[start:start + self.seq_len, :]
                    if np.isnan(unit.any()):
                        raise Exception('Found NaN in data')
                    data.append(unit)

        return data

    def reset_ptr(self):
        self.batch_ptr = 0

    def next_batch(self):
        """
        For absolute data: norm scale
        For relative data: raw scale(=norm_to_raw(norm_scale_rel_data))
        :return: dict('data'. 'rel_data')
        """
        # complete batch data
        if self.batch_ptr + self.batch_size <= self.count:
            batch_data = np.stack(self.data[self.batch_ptr:self.batch_ptr + self.batch_size], axis=0)
            self.batch_ptr += self.batch_size
        # incomplete batch data, left data not enough
        else:
            if self.batch_fragment:
                batch_data = np.stack(self.data[self.batch_ptr:], axis=0)
                self.batch_ptr = self.count
            else:
                raise Exception('No Complete Batch Data index = {} + batch_size = {} > count = {}'.format(
                    self.batch_ptr, self.batch_size, self.count
                ))
        batch_data = torch.from_numpy(batch_data).type(torch.float).to(self.device)
        rel_batch_data = abs_to_rel(self.norm_to_raw(batch_data))
        return {'data': batch_data, 'rel_data': rel_batch_data}

    def norm_to_raw(self, trajectory):
        trajectory = trajectory.clone().detach()
        loc_x_mean, loc_x_std = self.norm_metric['loc_x_mean'], self.norm_metric['loc_x_std']
        loc_z_mean, loc_z_std = self.norm_metric['loc_z_mean'], self.norm_metric['loc_z_std']
        if trajectory.shape[-1] == 5:
            trajectory[..., 0] = trajectory[..., 0] * loc_x_std + loc_x_mean
            trajectory[..., 1] = trajectory[..., 1] * loc_z_std + loc_z_mean
            trajectory[..., 2] = trajectory[..., 2] * loc_x_std
            trajectory[..., 3] = trajectory[..., 3] * loc_z_std
            return trajectory
        elif trajectory.shape[-1] == 2:
            trajectory[..., 0] = trajectory[..., 0] * loc_x_std + loc_x_mean
            trajectory[..., 1] = trajectory[..., 1] * loc_z_std + loc_z_mean
            return trajectory
        else:
            raise Exception('Not Recognized Data Shape.')

    def __len__(self):
        if not self.batch_fragment:
            return self.count // self.batch_size
        else:
            if self.count % self.batch_size != 0:
                return self.count // self.batch_size + 1
            else:
                return self.count // self.batch_size

    def get_mean_std(self):
        for target in self.norm_targets:
            self.norm_metric[target + '_mean'] = self.train_data[target].mean()
            self.norm_metric[target + '_std'] = self.train_data[target].std()
        self.recorder.logger.info('Norm Metric {}'.format(self.norm_metric))

    @staticmethod
    def rel_to_abs(y_hat, **kwargs):
        """
        Process model generated trajectories points.
            1. from relative to absolute.
        """
        post_y_hat = y_hat
        # post process relative to absolute
        if 'start' in kwargs.keys():
            post_y_hat = rel_to_abs(post_y_hat, kwargs['start'])
        else:
            raise Exception('Start location not specified.')
        return post_y_hat


if __name__ == '__main__':
    dl = SingleKittiDataLoader(file_path='../data/kitti-all-label02.csv',
                               batch_size=1,
                               trajectory_length=12,
                               device=torch.device('cpu'),
                               mode='train',
                               train_leave=[1],
                               recorder=None,
                               valid_scene=[1])
    x = dl.next_batch()
    print(x)
