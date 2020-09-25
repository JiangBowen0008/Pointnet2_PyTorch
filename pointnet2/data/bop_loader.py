import open3d as o3d

__all__ = [o3d]

import os
import os.path
import numpy as np
import json
from PIL import Image
import torch
import torch.utils.data as data

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def _get_data_files(list_filename):
    with open(list_filename) as f:
        return [line.rstrip() for line in f]


class BOPDataset(data.Dataset):
    def __init__(self, num_points, train=True, download=True, data_precent=1.0):
        super().__init__()
        self.data_precent = data_precent
        self.folder = "bop_sim"
        self.data_dir = os.path.join(BASE_DIR, self.folder)

        self.train, self.num_points = train, num_points

        # get data list path
        if train:
            data_list_path = os.path.join(self.data_dir, "train_list.txt")
        else:
            data_list_path = os.path.join(self.data_dir, "test_list.txt")
        self.data_list = _get_data_files(data_list_path)

        # get path of ground truth


def main():
    dset = BOPDataset(16, train=True)
    print(dset[0])
    print(len(dset))
    dloader = torch.utils.data.DataLoader(dset, batch_size=1, shuffle=True)
    for i, data in enumerate(dloader, 0):
        inputs, labels = data
        if i == len(dloader) - 1:
            print(inputs.size())


if __name__ == "__main__":
    main()
