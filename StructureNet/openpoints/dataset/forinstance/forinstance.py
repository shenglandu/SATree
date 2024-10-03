import os
import pickle
import logging
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from ..data_util import crop_pc, voxelize
from ..build import DATASETS
from ...utils.ply import *


@DATASETS.register_module()
class ForInstance(Dataset):
    classes = ['other', 'stem', 'crown']
    num_classes = 3
    num_per_class = np.array([2, 1, 3], dtype=np.int32)
    gravity_dim = 1

    def __init__(self,
                 data_root: str = '/mnt/data/Tree/FORinstance/dataset_ply',
                 voxel_size: float = 0.2,
                 voxel_max=None,
                 split: str = 'train',
                 transform=None,
                 loop: int = 1,
                 presample: bool = False,
                 variable: bool = False,
                 shuffle: bool = True,
                 ):
        super().__init__()

        self.split, self.voxel_size, self.transform, self.voxel_max, self.loop = \
            split, voxel_size, transform, voxel_max, loop
        self.presample = presample
        self.variable = variable
        self.shuffle = shuffle

        # obtain file list for the split
        self.raw_root = os.path.join(data_root, 'raw')
        self.train_root = os.path.join(data_root, 'train')
        self.val_root = os.path.join(data_root, 'val')
        # Du: later try splitting an extra val folder
        self.test_root = os.path.join(data_root, 'val')

        if split == 'train':
            # self.data_list = [self.scene_names[i] for i in self.train_split]
            self.data_list = []
            for f in os.listdir(self.train_root):
                f = f.split('.')[0]
                self.data_list.append(f)
        elif split == 'val':
            self.data_list = []
            for f in os.listdir(self.val_root):
                f = f.split('.')[0]
                self.data_list.append(f)
        elif split == 'test':
            self.data_list = []
            for f in os.listdir(self.test_root):
                f = f.split('.')[0]
                self.data_list.append(f)

        # pre-sample the validation data
        processed_root = os.path.join(data_root, 'processed')
        filename = os.path.join(processed_root, f'forinstance_{split}_{voxel_size:.3f}_{str(voxel_max)}.pkl')
        if presample and not os.path.exists(filename):
            if not split == 'val':
                pass
            np.random.seed(0)
            self.data = []
            for item in tqdm(self.data_list, desc=f'Loading ForInstanceFUll {split} split'):
                data_path = os.path.join(self.val_root, item + '.ply')
                cloud = read_ply(data_path)
                points = np.vstack((cloud['x'], cloud['y'], cloud['z'])).T
                points -= np.min(points, 0)
                # changed p to s for gaussian data
                i, s, labels, ins = np.expand_dims(cloud['i'], axis=1), \
                                    np.expand_dims(cloud['s'], axis=1), \
                                    np.expand_dims(cloud['class'], axis=1), \
                                    np.expand_dims(cloud['label'], axis=1)
                i = (2*i/np.max(i) - 1.0).astype(np.float32)
                s = s.astype(np.float32)
                if voxel_size:
                    uniq_idx = voxelize(points, voxel_size)
                    points, i, s, labels, ins = points[uniq_idx], i[uniq_idx], s[uniq_idx], labels[uniq_idx], ins[uniq_idx]
                    cdata = np.hstack((points, i, s, labels, ins))
                else:
                    cdata = np.hstack((points, i, s, labels, ins))
                self.data.append(cdata)
            npoints = np.array([len(data) for data in self.data])
            logging.info('split: %s, median npoints %.1f, avg num points %.1f, std %.1f'
                         % (self.split, np.median(npoints), np.average(npoints), np.std(npoints)))
            os.makedirs(processed_root, exist_ok=True)
            with open(filename, 'wb') as f:
                pickle.dump(self.data, f)
                print(f"{filename} saved successfully")
        elif presample:
            with open(filename, 'rb') as f:
                self.data = pickle.load(f)
                print(f"{filename} load successfully")

        # obtain information of the s=length of the data split
        self.data_idx = np.arange(len(self.data_list))
        assert len(self.data_idx) > 0
        logging.info(f"\nTotally {len(self.data_idx)} samples in {split} set")
    
    def get_offset(self, xyz, instance_label):
        pt_mean = np.ones((xyz.shape[0], 3), dtype=np.float32) * -100.0
        instance_num = max(int(instance_label.max()) + 1, 0)
        for i in range(instance_num):
            inst_idx_i = np.where(instance_label == i)
            xyz_i = xyz[inst_idx_i]
            pt_mean[inst_idx_i] = xyz_i.mean(0)
        pt_offset_label = pt_mean - xyz
        return pt_offset_label

    def __getitem__(self, idx):
        data_idx = self.data_idx[idx % len(self.data_idx)]

        # load data
        if self.split == 'val' and self.presample:
            points, i, s, labels, ins = np.split(self.data[data_idx], [3, 4, 5, 6], axis=1)
        else:
            if self.split == 'train':
                root = self.train_root
            else:
                root = self.test_root
            data_path = os.path.join(root, self.data_list[data_idx] + '.ply')
            cloud = read_ply(data_path)
            points = np.vstack((cloud['x'], cloud['y'], cloud['z'])).T
            points -= np.min(points, 0)
            features = np.vstack((cloud['i'], cloud['s'], cloud['label'])).T
            labels = np.expand_dims(cloud['class'], axis=1)
            points, features, labels = crop_pc(points, features, labels, self.split, self.voxel_size, self.voxel_max,
                                               downsample=not self.presample, variable=self.variable, shuffle=self.shuffle)
            i, s, ins = np.split(features, [1, 2], axis=1)
            i = (2 * i / np.max(i) - 1.0).astype(np.float32)
        labels = labels.squeeze(-1).astype(np.int64)
        s = s.squeeze(-1).astype(np.float32)
        ins = ins.squeeze(-1).astype(np.int64)
        data = {'pos': points, 'x': i, 'y': labels, 's': s, 'ins': ins}

        # augment data, add height feature
        if self.transform is not None:
            data = self.transform(data)
        if 'heights' not in data.keys():
            data['heights'] = torch.from_numpy(points[:, self.gravity_dim:self.gravity_dim + 1].astype(np.float32))

        transformed_xyz = data['pos'].numpy()
        data['offset'] = torch.from_numpy(self.get_offset(transformed_xyz, ins))

        return data

    def __len__(self):
        return len(self.data_idx) * self.loop
