import json
import torch
import numpy as np
import os
import random

from script.tools import rel_to_abs, abs_to_rel


class KittiDataLoader:
    def __init__(self, filepath, batch_size, trajectory_length, device, seed=17373321):
        self.count = 0
        self.batch_size = batch_size
        self.trajectory_length = trajectory_length
        self.batch_ptr = 0
        self.device = device

        if not os.path.exists(filepath):
            raise Exception('{} not exist'.format(filepath))
        with open(filepath, 'r') as f:
            self.objects = json.load(f)

        for o in self.objects:
            o.sort(key=lambda x: int(x['frame']))
            self.count += max(len(o) - trajectory_length + 1, 0)

        self.data = self.preprocess()
        random.seed(seed)
        random.shuffle(self.data, random=random.random)

        # print
        print('Count = {}, Batch Size = {}, Iteration = {}, Device = {}'.format(
            self.count, self.batch_size, self.__len__(), self.device))

    def preprocess(self):
        """
        process loaded data into list with length = count
        each unit in a list -> [trajectory_length, 2]
        """
        seq_ptr = 0
        obj_ptr = 0
        data = list()
        for t in range(self.count):
            # current object used up
            while seq_ptr + self.trajectory_length > len(self.objects[obj_ptr]):
                # update object pointer
                obj_ptr = obj_ptr + 1
                seq_ptr = 0

            o = self.objects[obj_ptr]
            unit_in_batch = np.zeros((self.trajectory_length, 2))
            for ptr in range(seq_ptr, seq_ptr + self.trajectory_length):
                unit_in_batch[ptr - seq_ptr][0] = o[ptr]['location'][0]
                unit_in_batch[ptr - seq_ptr][1] = o[ptr]['location'][2]

            seq_ptr += 1

            if np.isnan(unit_in_batch.any()):
                raise Exception('Found NaN in data.')

            data.append(unit_in_batch)
        return data

    def reset_ptr(self):
        self.batch_ptr = 0

    def next_batch(self):
        batch_data = np.stack(self.data[self.batch_ptr:self.batch_ptr + self.batch_size], axis=0)
        self.batch_ptr += self.batch_size
        batch_data = torch.from_numpy(batch_data).type(torch.float).to(self.device)
        rel_batch_data = abs_to_rel(batch_data)
        return {'data': batch_data, 'rel_data': rel_batch_data}

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

    def __len__(self):
        return self.count // self.batch_size


if __name__ == '__main__':
    loader = KittiDataLoader(os.path.join('..', 'data', 'kitti-train.json'), 1, 16)
    for i in range(5):
        print(loader.next_batch())
