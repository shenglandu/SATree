import numpy as np
import os
from os.path import getsize
from ply import *

scene_names = ['2023-01-09_5_1_37', '2023-01-09_17_2_18', '2023-01-09_tum_campus', '2023-01-10_7_6', '2023-01-10_11',
               '2023-01-10_38', '2023-01-10_46_13', '2023-01-10_47', '2023-01-11_51', '2023-01-12_27',
               '2023-01-12_28', '2023-01-12_29', '2023-01-12_30', '2023-01-12_35_34', '2023-01-12_48',
               '2023-01-12_55_53_54', '2023-01-12_56', '2023-01-12_57', '2023-01-12_58', '2023-01-12_59',
               '2023-01-12_65_64', '2023-01-13_4', '2023-01-13_19', '2023-01-13_20', '2023-01-13_40',
               '2023-01-13_41', '2023-01-13_42', '2023-01-13_52', '2023-01-13_61', '2023-01-13_70',
               '2023-01-13_74', '2023-01-16_3', '2023-01-16_12', '2023-01-16_16', '2023-01-16_21',
               '2023-01-16_22', '2023-01-16_23', '2023-01-16_24', '2023-01-16_43', '2023-01-16_44']
train_split = [0, 3, 4, 5, 8, 9, 10, 11, 12, 13, 15, 16, 18, 19, 20, 21,
               22, 23, 24, 26, 27, 28, 30, 31, 32, 34, 35, 36, 37, 38]
val_split = [1, 6, 14, 25, 33]
test_split = [2, 7, 17, 29, 39]


def crop_clouds(cloud, stride):
    cloud_max = np.amax(cloud[:, 0:3], axis=0)
    cloud_min = np.amin(cloud[:, 0:3], axis=0)
    width = int(np.ceil((cloud_max[0] - cloud_min[0]) / stride)) + 1
    depth = int(np.ceil((cloud_max[1] - cloud_min[1]) / stride)) + 1
    cells = [(i * stride, j * stride) for i in range(width) for j in range(depth)]

    blocks = []
    for (x, y) in cells:
        xcond = (cloud[:, 0] - cloud_min[0] <= x + stride) & (cloud[:, 0] - cloud_min[0] >= x)
        ycond = (cloud[:, 1] - cloud_min[1] <= y + stride) & (cloud[:, 1] - cloud_min[1] >= y)
        cond = xcond & ycond
        block = cloud[cond, :]
        blocks.append(block)

    return blocks


def prepare_files(files, out_path, crop_size=50):
    counter = 0
    thres_num = 1e5

    for file in files:
        cloud_name = os.path.basename(file).strip('.ply')
        data = read_ply(file)
        points = np.vstack((data['x'], data['y'], data['z'], data['i'], data['s'], data['class'], data['label'])).T
        blocks = crop_clouds(points, stride=crop_size)
        for bi, block in enumerate(blocks):
            if len(block) <= thres_num:
                counter += 1
                continue
            out_file = os.path.join(out_path, cloud_name + '_' + str(bi) + '.ply')
            write_ply(out_file, [block.astype(np.float32)], ['x', 'y', 'z', 'i', 's', 'class', 'label'])

        print(cloud_name + ' is finished\n')
    print('Total skipped file :%d' % counter)


if __name__ == '__main__':
    # Specify the path
    data_folder = '/mnt/data/Tree/TreeML-Data/Dataset_score_ply/'

    # # Make the training path
    # split = 'train'
    # train_path = os.path.join(data_folder, split)
    # os.makedirs(train_path, exist_ok=True)
    # train_files = [os.path.join(data_folder + 'raw/', scene_names[i] + '.ply') for i in train_split]
    # prepare_files(train_files, train_path, crop_size=50)
    #
    # # Make the validation path
    # split = 'val'
    # val_path = os.path.join(data_folder, split)
    # os.makedirs(val_path, exist_ok=True)
    # val_files = [os.path.join(data_folder + 'raw/', scene_names[i] + '.ply') for i in val_split]
    # prepare_files(val_files, val_path, crop_size=50)

    # Make the validation path
    split = 'test'
    test_path = os.path.join(data_folder, split)
    os.makedirs(test_path, exist_ok=True)
    test_files = [os.path.join(data_folder + 'raw/', scene_names[i] + '.ply') for i in test_split]
    prepare_files(test_files, test_path, crop_size=100)
