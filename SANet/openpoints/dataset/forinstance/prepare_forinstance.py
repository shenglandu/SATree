# Author: Shenglan Du
# Time: 01-10-2024


from os import makedirs
from os.path import exists, join, isfile, basename
import numpy as np
import ply
import pandas as pd
import laspy


semantic_name_to_class = {'ignore': -1,
                          'other': 0,
                          'stem': 1,
                          'crown': 2}
background_label = -100


def get_gaussian_heatmap(points, stem_filter, scale=10):
    """
    Obtain the main crown points
        points: points of a tree instance
        stem_filter: the stem mask to obtain stem points
        scale: the Gaussian heatmap scale
    Return:
        scores: the GT Gaussian heat scores
    """
    stem_points = points[stem_filter]
    root = np.expand_dims(np.mean(stem_points, axis=0), axis=0)
    offsets = points - root
    dists = np.sum(offsets[:, :2] ** 2, axis=1)
    gaussian_scores = np.exp(-scale * (dists / dists.max() + 1e-6))

    # Enhance gaussian scores at stems
    gaussian_scores[stem_filter] = 1
    return gaussian_scores


def crop_clouds(cloud, stride):
    """
    Crop the input cloud into square blocks of clouds
        cloud: the input point cloud
        stride: the cropping size
    Return:
        block: the blocks of clouds
    """
    cloud_max = np.amax(cloud[:, 0:3], axis=0)
    cloud_min = np.amin(cloud[:, 0:3], axis=0)
    width = int(np.ceil((cloud_max[0] - cloud_min[0]) / stride)) + 1
    depth = int(np.ceil((cloud_max[1] - cloud_min[1]) / stride)) + 1
    cells = [(i * stride, j * stride) for i in range(width) for j in range(depth)]

    # Retrieve over blocks
    blocks = []
    for (x, y) in cells:
        xcond = (cloud[:, 0] - cloud_min[0] <= x + stride) & (cloud[:, 0] - cloud_min[0] >= x)
        ycond = (cloud[:, 1] - cloud_min[1] <= y + stride) & (cloud[:, 1] - cloud_min[1] >= y)
        cond = xcond & ycond
        block = cloud[cond, :]
        blocks.append(block)

    return blocks


def prepare_files(files, out_path, crop_size=50, split='train'):
    """
    Prepare the files for network training and validation
        files: the file list
        out_path: the path to store output files
        split: the data split
    """
    counter = 0
    thres_num = 1e5

    for file in files:
        cloud_name = basename(file).strip('.ply')
        data = ply.read_ply(file)
        points = np.vstack((data['x'], data['y'], data['z'], data['i'], data['s'], data['class'], data['label'])).T
        if split == 'train':
            points = points[points[:, 5] != semantic_name_to_class['ignore']]
        blocks = crop_clouds(points, stride=crop_size)
        for bi, block in enumerate(blocks):
            if len(block) <= thres_num:
                counter += 1
                continue
            out_file = join(out_path, cloud_name + '_' + str(bi) + '.ply')
            ply.write_ply(out_file, [block.astype(np.float32)], ['x', 'y', 'z', 'i', 's', 'class', 'label'])
    print('Total skipped file :%d' % counter)


def prepare_scene_clouds(tree_path, trainval_split_path, process_path):
    """
    Prepare the scene clouds
        tree_path: path to tree data
        trainval_split_path: path to the train val split file
        process_path: path to write the scene clouds as ply
    """
    # Read the tree data file
    df = pd.read_csv(trainval_split_path)
    train_files = []
    val_files = []

    for idx, row in df.iterrows():
        file_path = join(tree_path, row['path'])
        split = row['split']
        cloud_name = row['path'].split('/')[0] + '_' + (row['path'].split('/')[1]).split('.')[0]
        print('Processing scene - ' + cloud_name)
        if not isfile(file_path):
            print('no file exists for this scene')
            continue

        out_file = join(process_path, cloud_name + '.ply')
        if split == 'dev':
            train_files.append(out_file)
        else:
            val_files.append(out_file)
        if isfile(out_file):
            print('The scene has been processed')
            continue

        # read points with attributes
        las_clouds = laspy.read(file_path)
        cloud_points = np.vstack((las_clouds.x, las_clouds.y, las_clouds.z)).transpose()
        offsets = np.asarray(las_clouds.header.offsets)
        cloud_points -= offsets
        cloud_intensities = np.asarray(las_clouds.intensity)
        cloud_classes = np.asarray(las_clouds.classification).astype(np.int32)
        cloud_labels = np.asarray(las_clouds.treeID).astype(np.int32)

        # remap cloud class code
        ignore_filter = np.logical_or(cloud_classes == 0, cloud_classes == 3)
        cloud_classes[ignore_filter] = semantic_name_to_class['ignore']
        other_filter = np.logical_or(cloud_classes == 1, cloud_classes == 2)
        cloud_classes[other_filter] = semantic_name_to_class['other']
        cloud_classes[cloud_classes == 4] = semantic_name_to_class['stem']
        cloud_classes[cloud_classes >= 5] = semantic_name_to_class['crown']

        # remap cloud instance label code
        cloud_scores = np.zeros((cloud_points.shape[0], ))
        new_cloud_labels = np.zeros_like(cloud_labels)
        unique_labels = np.unique(cloud_labels)
        ins_id = 0
        for i in unique_labels:
            if i == 0:
                new_cloud_labels[cloud_labels == i] = background_label
                # fix the margin tree labelling bug in rmit and tuwien
                tree_mask = (cloud_classes == semantic_name_to_class['stem']) | (cloud_classes == semantic_name_to_class['crown'])
                overlap_mask = (cloud_labels == i) & tree_mask
                cloud_classes[overlap_mask] = semantic_name_to_class['ignore']
            else:
                new_cloud_labels[cloud_labels == i] = ins_id
                ins_id += 1
                # calculate gaussian heat scores per tree instance
                tree_i = cloud_points[cloud_labels == i]
                class_i = cloud_classes[cloud_labels == i]
                stem_filter = class_i == semantic_name_to_class['stem']
                score_i = get_gaussian_heatmap(tree_i, stem_filter)
                cloud_scores[cloud_labels == i] = score_i

        ply.write_ply(out_file,
                      [cloud_points, cloud_intensities, cloud_scores, cloud_classes, new_cloud_labels],
                      ['x', 'y', 'z', 'i', 's', 'class', 'label'])
        del cloud_points, cloud_intensities, cloud_scores, cloud_classes, cloud_labels, new_cloud_labels

    return train_files, val_files


if __name__ == '__main__':
    # Specify the paths to data
    tree_path = '/mnt/data/Tree/FORinstance/dataset_tree'
    trainval_split_path = join(tree_path, 'data_split_metadata.csv')

    # Specify the processed data path
    root_path = '/mnt/data/Tree/FORinstance/dataset_ply'
    process_path = join(root_path, 'raw')
    if not exists(process_path):
        makedirs(process_path)

    print('-------------------------------')
    print('Prepare scene clouds as ply files\n')
    train_files, val_files = prepare_scene_clouds(tree_path, trainval_split_path, process_path)

    print('-------------------------------')
    print('Cut scene clouds into patches\n')
    # Make the training path
    split = 'train'
    train_path = join(root_path, split)
    makedirs(train_path, exist_ok=True)
    prepare_files(train_files, train_path, crop_size=20, split='train')

    # Make the validation path
    split = 'val'
    val_path = join(root_path, split)
    makedirs(val_path, exist_ok=True)
    prepare_files(val_files, val_path, crop_size=150, split='val')
