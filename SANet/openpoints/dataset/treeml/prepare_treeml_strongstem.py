# Author: Shenglan Du
# Time: 31-05-2024


from os import listdir, makedirs
from os.path import exists, join, isfile, basename
import numpy as np
import ply
import pandas as pd
from sklearn.neighbors import KDTree
import h5py


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
semantic_name_to_class = {'other': 0,
                          'stem': 1,
                          'crown': 2}
background_label = -100


def get_structure_from_qsm(file_qsm, file_trans, points, eps_d=0.8, scale=10):
    """
    Obtain the main crown points
        file_qsm: the file that stores qsm graph
        file_trans: the file that stores the transpose coordinates
        points: the 3d points of the tree
        stem_filter: the stem mask to obtain stem points
    Return:
        potential: the point-wise tree growth potential
        noise_filter: the mask of noisy outlier
    """
    # Read the trans file
    with h5py.File(file_trans, 'r') as f:
        key = list(f.keys())[0]
        trans_array = f[key]
        x_trans = trans_array[0][0]
        y_trans = trans_array[1][0]
        z_trans = trans_array[2][0]
    trans_array = np.array([[x_trans, y_trans, z_trans]]).astype(np.float32)

    # Read the qsm file and get skeleton points
    df = pd.read_csv(file_qsm)

    # Get branch start coordinates and branch orders
    vertices = np.column_stack((df['start_x'], df['start_y'], df['start_z']))
    vertices += trans_array
    branch_IDs = df['branchID'].to_numpy()
    stem_indicator = np.where(branch_IDs == 1, branch_IDs, 0)

    # Project the potential to the original cloud and track the stem points
    stem_filter = np.zeros((points.shape[0], ))
    kd = KDTree(vertices, leaf_size=50)
    noise_filter = []
    for i in range(len(points)):
        dist, idx = kd.query(points[i:i+1], k=1)
        dist, idx = dist[0][0], idx[0][0]
        if dist <= eps_d:
            stem_filter[i] = stem_indicator[idx]
        else:
            noise_filter.append(i)
    stem_filter = stem_filter.astype(bool)

    # Compute gaussian scores where stem points have a score of 1
    stem_points = points[stem_filter]
    root = np.expand_dims(np.mean(stem_points, axis=0), axis=0)
    offsets = points - root
    dists = np.sum(offsets[:, :2] ** 2, axis=1)
    dists[noise_filter] = 0.
    gaussian_scores = np.exp(-scale * (dists / dists.max() + 1e-6))
    gaussian_scores[stem_filter] = 1.
    gaussian_scores[noise_filter] = 0.

    return gaussian_scores, stem_filter, noise_filter


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


def prepare_files(files, out_path, crop_size=50, thres_num=1e5):
    """
    Prepare the files for network training and validation
        files: the file list
        out_path: the path to store output files
        split: the data split
    """
    counter = 0
    for file in files:
        cloud_name = basename(file).strip('.ply')
        data = ply.read_ply(file)
        points = np.vstack((data['x'], data['y'], data['z'], data['i'], data['s'], data['class'], data['label'])).T
        blocks = crop_clouds(points, stride=crop_size)
        for bi, block in enumerate(blocks):
            if len(block) <= thres_num:
                counter += 1
                continue
            out_file = join(out_path, cloud_name + '_' + str(bi) + '.ply')
            ply.write_ply(out_file, [block.astype(np.float32)], ['x', 'y', 'z', 'i', 's', 'class', 'label'])
    print('Total skipped file :%d' % counter)


def prepare_scene_clouds(tree_path,
                         other_path,
                         qsm_path,
                         process_path):
    """
    Prepare the scene clouds
        tree_path: path to tree data
        other_path: path to rest points in the scene
        qsm_path: path to qsm tree graphs
        process_path: path to write the scene clouds as ply
    """
    for i, scene in enumerate(scene_names):
        print('Processing scene - ' + scene)
        # Specify the folders
        tree_folder = join(tree_path, scene)
        other_folder = join(other_path, scene)
        tree_qsm_folder = join(qsm_path, scene)

        # Check if the scene has been processed or not
        out_name = scene + '.ply'
        out_file = join(process_path, out_name)
        if isfile(out_file):
            print('The scene has been processed')
            continue

        # Initialize the point cloud containers
        cloud_points = np.empty((0, 3), dtype=np.float32)
        cloud_intensities = np.empty((0, 1), dtype=np.float32)
        cloud_classes = np.empty((0, 1), dtype=np.int32)
        cloud_labels = np.empty((0, 1), dtype=np.int32)
        cloud_scores = np.empty((0, 1), dtype=np.float32)

        # Add building and other points
        print('Others...')
        flag = False
        for file_name in listdir(other_folder):
            # fix the scene problem of 2023-01-13_40
            if 'label' in file_name:
                print('The current scene needs to ignore building points')
                flag = True

        for file_name in listdir(other_folder):
            # fix the scene problem of 2023-01-13_40, 2023-01-12_30
            if flag and 'building' in file_name:
                continue
            if file_name.split('.')[1] == 'txt':
                other_file = join(other_folder, file_name)
                other_data = np.loadtxt(other_file)
                # filter out non-tree points for this scene
                if flag:
                    other_data = other_data[other_data[:, 4] < 2]
                other_classes = np.zeros((other_data.shape[0], 1), dtype=np.int32) + semantic_name_to_class['other']
                other_labels = np.zeros((other_data.shape[0], 1), dtype=np.int32) + background_label
                other_scores = np.zeros((other_data.shape[0], 1), dtype=np.float32)
                # Add data to the containers
                cloud_points = np.vstack((cloud_points, other_data[:, 0:3].astype(np.float32)))
                cloud_intensities = np.vstack((cloud_intensities, other_data[:, [3]].astype(np.float32)))
                cloud_classes = np.vstack((cloud_classes, other_classes))
                cloud_labels = np.vstack((cloud_labels, other_labels))
                cloud_scores = np.vstack((cloud_scores, other_scores))
                # clean memories
                del other_data, other_classes, other_labels, other_scores

        # Add tree points
        for file_name in listdir(tree_folder):
            if file_name.split('.')[1] == 'txt':
                if 'trashtree' in file_name:
                    continue
                # Read tree points and id
                tree_id = file_name.split('.')[0]
                ins_id = int(tree_id.split('_')[-1])
                tree_file = join(tree_folder, file_name)

                print('Reading tree - %s' % (ins_id))
                tree_data = np.loadtxt(tree_file)
                tree_points = tree_data[:, :3]

                # Get structure from tree qsm
                qsm_fname = 'OptQSM_' + tree_id + '.csv'
                trans_fname = 'trans_' + tree_id + '.mat'
                file_qsm = join(tree_qsm_folder, 'optcsv', qsm_fname)
                file_trans = join(tree_qsm_folder, 'trans', trans_fname)
                if isfile(file_qsm) and isfile(file_trans):
                    scores, stem_filter, noise_filter = get_structure_from_qsm(file_qsm, file_trans, tree_points)
                else:
                    print(str(tree_id) + ': qsm does not exist!')
                    continue

                tree_classes = np.zeros((tree_points.shape[0], 1), dtype=np.int32) + semantic_name_to_class['crown']
                tree_classes[stem_filter] = semantic_name_to_class['stem']
                tree_classes[noise_filter] = semantic_name_to_class['other']
                tree_labels = np.zeros((tree_data.shape[0], 1), dtype=np.int32) + ins_id
                tree_labels[noise_filter] = background_label
                # Add data to the containers
                cloud_points = np.vstack((cloud_points, tree_data[:, :3].astype(np.float32)))
                cloud_intensities = np.vstack((cloud_intensities, tree_data[:, [3]].astype(np.float32)))
                cloud_classes = np.vstack((cloud_classes, tree_classes.astype(np.int32)))
                cloud_labels = np.vstack((cloud_labels, tree_labels.astype(np.int32)))
                cloud_scores = np.vstack((cloud_scores, np.expand_dims(scores, axis=1)))

                del tree_data, tree_points, tree_labels, tree_classes, scores

        # Save as ply
        ply.write_ply(out_file,
                     [cloud_points, cloud_intensities, cloud_scores, cloud_classes, cloud_labels],
                     ['x', 'y', 'z', 'i', 's', 'class', 'label'])
        del cloud_points, cloud_intensities, cloud_scores, cloud_classes, cloud_labels


if __name__ == '__main__':
    # Specify the paths to data
    tree_path = '/mnt/data/Tree/TreeML-Data/Dataset_tree'
    building_other_path = '/mnt/data/Tree/TreeML-Data/Dataset_building_other/Dataset_segmentation'
    tree_qsm_path = '/mnt/data/Tree/TreeML-Data/Dataset_QSM/'

    # Specify the processed data path
    root_path = '/mnt/data/Tree/TreeML-Data/Dataset_strstem_ply'
    process_path = join(root_path, 'raw')
    if not exists(process_path):
        makedirs(process_path)

    print('-------------------------------')
    print('Prepare scene clouds as ply files\n')
    prepare_scene_clouds(tree_path, building_other_path, tree_qsm_path, process_path)

    print('-------------------------------')
    print('Cut scene clouds into patches\n')
    # Make the training path
    split = 'train'
    train_path = join(root_path, split)
    makedirs(train_path, exist_ok=True)
    train_files = [join(process_path, scene_names[i] + '.ply') for i in train_split]
    prepare_files(train_files, train_path, crop_size=50)

    # Make the validation path
    split = 'val'
    val_path = join(root_path, split)
    makedirs(val_path, exist_ok=True)
    val_files = [join(process_path, scene_names[i] + '.ply') for i in val_split]
    prepare_files(val_files, val_path, crop_size=50)

    # Make the test path
    split = 'test'
    test_path = join(root_path, split)
    makedirs(test_path, exist_ok=True)
    test_files = [join(process_path, scene_names[i] + '.ply') for i in test_split]
    prepare_files(test_files, test_path, crop_size=100)
