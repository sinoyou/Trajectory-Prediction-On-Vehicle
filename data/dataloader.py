import json
import torch
import numpy as np
import os


class KittiDataLoader:
    def __init__(self, filepath, batch_size, trajectory_length):
        self.count = 0
        self.batch_size = batch_size
        self.trajectory_length = trajectory_length

        self.object_ptr = 0
        self.seq_ptr = 0
        self.object_cache = None

        if not os.path.exists(filepath):
            raise Exception('{} not exist'.format(filepath))
        with open(filepath, 'r') as f:
            self.objects = json.load(f)
        for o in self.objects:
            o.sort(key=lambda x: int(x['frame']))
            self.count += len(o) - trajectory_length + 1

    def reset_ptr(self):
        self.object_ptr = 0
        self.seq_ptr = 0
        self.object_cache = None

    def next_batch(self):
        tensor = np.zeros((self.batch_size, self.trajectory_length, 2))
        for i in range(self.batch_size):
            # current object used up
            if self.seq_ptr + self.trajectory_length > len(self.objects[self.object_ptr]):
                # update object pointer
                self.object_ptr = (self.object_ptr + 1) % self.__len__()
                self.seq_ptr = 0
                self.object_cache = None

            # set cache
            if self.object_cache is None:
                o = self.objects[self.object_ptr]
                self.object_cache = np.zeros((len(o), 2))
                for t in range(len(o)):
                    self.object_cache[t][0] = o[t]['location'][2]
                    self.object_cache[t][1] = o[t]['location'][0]

            tensor[i, :, :] = self.object_cache[self.seq_ptr: self.seq_ptr + self.trajectory_length, :]
            self.seq_ptr += 1

        return torch.from_numpy(tensor)

    def __len__(self):
        return self.count // self.batch_size


if __name__ == '__main__':
    loader = KittiDataLoader(os.path.join('..', 'save', 'kitti-train.json'), 1, 5)
    for i in range(5):
        print(loader.next_batch())