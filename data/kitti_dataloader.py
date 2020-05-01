import torch
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

    def __init__(self, file_path, batch_size, trajectory_length, device, seed=17373321, leave_scene=None,
                 valid_scene=None):
        self.count = 0
        self.batch_size = batch_size
        self.seq_len = trajectory_length
        self.batch_ptr = 0
        self.device = device

        # check args
        if leave_scene is None and valid_scene is None:
            raise Exception('Invalid Args')

        # read raw data. 
        raw_data = pd.read_csv(file_path)
        self.loc_x_mean = raw_data['loc_x_mean'].unique()[0]
        self.loc_x_std = raw_data['loc_x_std'].unique()[0]
        self.loc_z_mean = raw_data['loc_z_mean'].unique()[0]
        self.loc_z_std = raw_data['loc_z_std'].unique()[0]

        # filter scene
        if leave_scene:
            # traing scene
            print('Scenes {} are leaved.'.format(leave_scene))
            leaves = [raw_data['scene'] == s for s in leave_scene]
            mask = leaves[0]
            for i in range(1, len(leaves)):
                mask = mask | leaves[i]
            raw_data = raw_data[~mask]
        elif valid_scene:
            # valid scene
            print('Scenes {} are used for validation.'.format(valid_scene))
            targets = [raw_data['scene'] == s for s in valid_scene]
            mask = targets[0]
            for i in range(1, len(targets)):
                mask = mask | targets[i]
            raw_data = raw_data[mask]
        else:
            raise Exception('Invalid Args')

        self.data = self.preprocess(raw_data)
        self.count = len(self.data)
        random.seed(seed)
        random.shuffle(self.data, random=random.random)

        # print summary. 
        print('Count = {}, Batch Size = {}, Iteration = {}, Device = {}'.format(
            self.count, self.batch_size, self.__len__(), self.device
        ))

    def preprocess(self, raw):
        """
        process loaded data into list with length = count
        each unit in a list -> [trajectory_length, 2]
        """
        data = list()
        scenes = raw['scene'].unique()
        for scene in scenes:
            raw_scene = raw[raw['scene'] == scene]
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

    def reset_prt(self):
        self.batch_ptr = 0

    def next_batch(self):
        batch_data = np.stack(self.data[self.batch_ptr:self.batch_ptr + self.batch_size], axis=0)
        self.batch_ptr += self.batch_size
        batch_data = torch.from_numpy(batch_data).type(torch.float).to(self.device)
        rel_batch_data = abs_to_rel(batch_data)
        return {'data': batch_data, 'rel_data': rel_batch_data}

    def norm_to_raw(self, trajectory):
        if trajectory.shape[-1] == 5:
            trajectory[..., 0] = trajectory[..., 0] * self.loc_x_std + self.loc_x_mean
            trajectory[..., 1] = trajectory[..., 1] * self.loc_z_std + self.loc_z_mean
            trajectory[..., 2] = trajectory[..., 0] * self.loc_x_std
            trajectory[..., 3] = trajectory[..., 0] * self.loc_z_std
            return trajectory
        elif trajectory.shape[-1] == 2:
            trajectory[..., 0] = trajectory[..., 0] * self.loc_x_std + self.loc_x_mean
            trajectory[..., 1] = trajectory[..., 1] * self.loc_z_std + self.loc_z_mean
            return trajectory
        else:
            raise Exception('Not Recognized Data Shape.')

    def __len__(self):
        return self.count // self.batch_size

    @staticmethod
    def rel_to_abs(y_hat, **kwargs):
        """
        Process model generated trajectories points.
            1. from relative to absolute.
        :param y_hat:
        :param kwargs:
        :return:
        """
        post_y_hat = y_hat
        # post process relative to absolute
        if 'start' in kwargs.keys():
            post_y_hat = rel_to_abs(post_y_hat, kwargs['start'])
        return post_y_hat


if __name__ == '__main__':
    dl = SingleKittiDataLoader('kitti-all-label02.csv', batch_size=4,
                               trajectory_length=12, device=torch.device('cpu'), leave_scene=[0])
