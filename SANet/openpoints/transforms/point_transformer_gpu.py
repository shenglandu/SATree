# Copyright (c) Chris Choy (chrischoy@ai.stanford.edu). All Rights Reserved.
#
# Please cite "4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural
# Networks", CVPR'19 (https://arxiv.org/abs/1904.08755) if you use any part of
# the code.
import random, logging
import numpy as np
import torch
import collections
from .transforms_factory import DataTransforms
#
# import scipy
# import scipy.ndimage
# import scipy.interpolate
from scipy.linalg import expm, norm


@DataTransforms.register_module()
class PointCloudToTensor(object):
    def __init__(self, **kwargs):
        pass

    def __call__(self, data):
        pts = data['pos']
        normals = data['normals'] if 'normals' in data.keys() else None
        colors = data['colors'] if 'colors' in data.keys() else None
        data['pos'] = torch.from_numpy(pts).float()
        if normals is not None:
            data['normals'] = torch.from_numpy(normals).float().transpose(0, 1)
        if colors is not None:
            data['colors'] = torch.from_numpy(colors).transpose(0, 1).float()
        return data


@DataTransforms.register_module()
class PointCloudCenterAndNormalize(object):
    def __init__(self, centering=True,
                 normalize=True,
                 gravity_dim=2,
                 append_xyz=False,
                 **kwargs):
        self.centering = centering
        self.normalize = normalize
        self.gravity_dim = gravity_dim
        self.append_xyz = append_xyz

    def __call__(self, data):
        if hasattr(data, 'keys'):
            if self.append_xyz:
                data['heights'] = data['pos'] - torch.min(data['pos'])
            else:
                height = data['pos'][:, self.gravity_dim:self.gravity_dim + 1]
                data['heights'] = height - torch.min(height)

            if self.centering:
                data['pos'] = data['pos'] - torch.mean(data['pos'], axis=0, keepdims=True)

            if self.normalize:
                m = torch.max(torch.sqrt(torch.sum(data['pos'] ** 2, axis=-1, keepdims=True)), axis=0, keepdims=True)[0]
                data['pos'] = data['pos'] / m
        else:
            if self.centering:
                data = data - torch.mean(data, axis=-1, keepdims=True)
            if self.normalize:
                m = torch.max(torch.sqrt(torch.sum(data ** 2, axis=-1, keepdims=True)), axis=0, keepdims=True)[0]
                data = data / m
        return data


@DataTransforms.register_module()
class PointCloudXYZAlign(object):
    """Centering the point cloud in the xy plane
    Args:
        object (_type_): _description_
    """

    def __init__(self,
                 gravity_dim=2,
                 **kwargs):
        self.gravity_dim = gravity_dim

    def __call__(self, data):
        if hasattr(data, 'keys'):
            data['pos'] -= torch.mean(data['pos'], axis=0, keepdims=True)
            data['pos'][:, self.gravity_dim] -= torch.min(data['pos'][:, self.gravity_dim])
        else:
            data -= torch.mean(data, axis=0, keepdims=True)
            data[:, self.gravity_dim] -= torch.min(data[:, self.gravity_dim])
        return data


@DataTransforms.register_module()
class RandomDropout(object):
    def __init__(self, dropout_ratio=0.2, dropout_application_ratio=0.2, **kwargs):
        """
        upright_axis: axis index among x,y,z, i.e. 2 for z
        """
        self.dropout_ratio = dropout_ratio
        self.dropout_application_ratio = dropout_application_ratio

    def __call__(self, data):
        if random.random() < self.dropout_application_ratio:
            N = len(data['pos'])
            inds = torch.randperm(N)[:int(N * (1 - self.dropout_ratio))]
            for k, v in data.items():
                if isinstance(v, torch.Tensor):
                    data[k] = v[inds]
        return data


@DataTransforms.register_module()
class RandomHorizontalFlip(object):
    def __init__(self, upright_axis, aug_prob=0.95, **kwargs):
        """
        upright_axis: axis index among x,y,z, i.e. 2 for z
        """
        self.D = 3
        self.upright_axis = {'x': 0, 'y': 1, 'z': 2}[upright_axis.lower()]
        # Use the rest of axes for flipping.
        self.horz_axes = set(range(self.D)) - set([self.upright_axis])
        self.aug_prob = aug_prob

    def __call__(self, data):
        if random.random() < self.aug_prob:
            for curr_ax in self.horz_axes:
                if random.random() < 0.5:
                    coord_max = torch.max(data['pos'])
                    data['pos'][:, curr_ax] = coord_max - data['pos'][:, curr_ax]
                    if 'normals' in data:
                        data['normals'][:, curr_ax] = -data['normals'][:, curr_ax]

        return data


@DataTransforms.register_module()
class PointCloudScaling(object):
    def __init__(self,
                 scale=[2. / 3, 3. / 2],
                 anisotropic=True,
                 scale_xyz=[True, True, True],
                 mirror=[0, 0, 0],  # the possibility of mirroring. set to a negative value to not mirror
                 **kwargs):
        self.scale_min, self.scale_max = np.array(scale).astype(np.float32)
        self.anisotropic = anisotropic
        self.scale_xyz = scale_xyz
        self.mirror = torch.from_numpy(np.array(mirror))
        self.use_mirroring = torch.sum(torch.tensor(self.mirror)>0) != 0

    def __call__(self, data):
        device = data['pos'].device if hasattr(data, 'keys') else data.device
        scale = torch.rand(3 if self.anisotropic else 1, dtype=torch.float32, device=device) * (
                self.scale_max - self.scale_min) + self.scale_min
        if self.use_mirroring:
            assert self.anisotropic==True
            self.mirror = self.mirror.to(device)
            mirror = (torch.rand(3, device=device) > self.mirror).to(torch.float32) * 2 - 1
            scale *= mirror
        for i, s in enumerate(self.scale_xyz):
            if not s: scale[i] = 1
        if hasattr(data, 'keys'):
            data['pos'] *= scale
        else:
            data *= scale
        return data


@DataTransforms.register_module()
class PointCloudTranslation(object):
    def __init__(self, shift=[0.2, 0.2, 0.], **kwargs):
        self.shift = torch.from_numpy(np.array(shift)).to(torch.float32)

    def __call__(self, data):
        device = data['pos'].device if hasattr(data, 'keys') else data.device
        translation = torch.rand(3, dtype=torch.float32, device=device) * self.shift.to(device)
        if hasattr(data, 'keys'):
            data['pos'] += translation
        else:
            data += translation
        return data


@DataTransforms.register_module()
class PointCloudScaleAndTranslate(object):
    def __init__(self, scale=[2. / 3, 3. / 2], scale_xyz=[True, True, True],  # ratio for xyz dimenions
                 anisotropic=True,
                 shift=[0.2, 0.2, 0.2],
                 mirror=[0, 0, 0],  # the possibility of mirroring. set to a negative value to not mirror
                 **kwargs):
        self.scale_min, self.scale_max = np.array(scale).astype(np.float32)
        self.shift = torch.from_numpy(np.array(shift)).to(torch.float32)
        self.scale_xyz = scale_xyz
        self.anisotropic = anisotropic
        self.mirror = torch.from_numpy(np.array(mirror))
        self.use_mirroring = torch.sum(self.mirror>0) != 0

    def __call__(self, data):
        device = data['pos'].device if hasattr(data, 'keys') else data.device
        scale = torch.rand(3 if self.anisotropic else 1, dtype=torch.float32, device=device) * (
                self.scale_max - self.scale_min) + self.scale_min
        # * note : scale_xyz has higher priority than mirror
        if self.use_mirroring:
            assert self.anisotropic==True
            self.mirror = self.mirror.to(device)
            mirror = (torch.rand(3, device=device) > self.mirror).to(torch.float32) * 2 - 1
            scale *= mirror
        for i, s in enumerate(self.scale_xyz):
            if not s: scale[i] = 1
        translation = (torch.rand(3, dtype=torch.float32, device=device) - 0.5) * 2 * self.shift.to(device)
        if hasattr(data, 'keys'):
            data['pos'] = torch.mul(data['pos'], scale) + translation
        else:
            data = torch.mul(data, scale) + translation
        return data


@DataTransforms.register_module()
class PointCloudJitter(object):
    def __init__(self, jitter_sigma=0.01, jitter_clip=0.05, **kwargs):
        self.noise_std = jitter_sigma
        self.noise_clip = jitter_clip

    def __call__(self, data):
        if hasattr(data, 'keys'):
            noise = torch.randn_like(data['pos']) * self.noise_std
            data['pos'] += noise.clamp_(-self.noise_clip, self.noise_clip)
        else:
            noise = torch.randn_like(data) * self.noise_std
            data += noise.clamp_(-self.noise_clip, self.noise_clip)
        return data


@DataTransforms.register_module()
class PointCloudScaleAndJitter(object):
    def __init__(self,
                 scale=[2. / 3, 3. / 2],
                 scale_xyz=[True, True, True],  # ratio for xyz dimenions
                 anisotropic=True,  # scaling in different ratios for x, y, z
                 jitter_sigma=0.01, jitter_clip=0.05,
                 mirror=[0, 0, 0],  # mirror scaling, x --> -x
                 **kwargs):
        self.scale_min, self.scale_max = np.array(scale).astype(np.float32)
        self.scale_xyz = scale_xyz
        self.noise_std = jitter_sigma
        self.noise_clip = jitter_clip
        self.anisotropic = anisotropic
        self.mirror = torch.from_numpy(np.array(mirror))

    def __call__(self, data):
        device = data['pos'].device if hasattr(data, 'keys') else data.device
        scale = torch.rand(3 if self.anisotropic else 1, dtype=torch.float32, device=device) * (
                self.scale_max - self.scale_min) + self.scale_min
        mirror = torch.round(torch.rand(3, device=device)) * 2 - 1
        self.mirror = self.mirror.to(device)
        mirror = mirror * self.mirror + (1 - self.mirror)
        scale *= mirror
        for i, s in enumerate(self.scale_xyz):
            if not s: scale[i] = 1
        if hasattr(data, 'keys'):
            noise = (torch.randn_like(data['pos']) * self.noise_std).clamp_(-self.noise_clip, self.noise_clip)
            data['pos'] = torch.mul(data['pos'], scale) + noise
        else:
            noise = (torch.randn_like(data) * self.noise_std).clamp_(-self.noise_clip, self.noise_clip)
            data = torch.mul(data, scale) + noise
        return data


@DataTransforms.register_module()
class PointCloudRotation(object):
    def __init__(self, angle=[0, 0, 0], **kwargs):
        self.angle = np.array(angle) * np.pi

    @staticmethod
    def M(axis, theta):
        return expm(np.cross(np.eye(3), axis / norm(axis) * theta))

    def __call__(self, data):
        if hasattr(data, 'keys'):
            device = data['pos'].device
        else:
            device = data.device

        if isinstance(self.angle, collections.Iterable):
            rot_mats = []
            for axis_ind, rot_bound in enumerate(self.angle):
                theta = 0
                axis = np.zeros(3)
                axis[axis_ind] = 1
                if rot_bound is not None:
                    theta = np.random.uniform(-rot_bound, rot_bound)
                rot_mats.append(self.M(axis, theta))
            # Use random order
            np.random.shuffle(rot_mats)
            rot_mat = torch.tensor(rot_mats[0] @ rot_mats[1] @ rot_mats[2], dtype=torch.float32, device=device)
        else:
            raise ValueError()

        """ DEBUG
        from openpoints.dataset import vis_multi_points
        old_points = data.cpu().numpy()
        # old_points = data['pos'].numpy()
        # new_points = (data['pos'] @ rot_mat.T).numpy()
        new_points = (data @ rot_mat.T).cpu().numpy()
        vis_multi_points([old_points, new_points])
        End of DEBUG"""

        if hasattr(data, 'keys'):
            data['pos'] = data['pos'] @ rot_mat.T
            if 'normals' in data:
                data['normals'] = data['normals'] @ rot_mat.T
        else:
            data = data @ rot_mat.T
        return data


@DataTransforms.register_module()
class ChromaticDropGPU(object):
    def __init__(self, color_drop=0.2, **kwargs):
        self.color_drop = color_drop

    def __call__(self, data):
        if torch.rand(1) < self.color_drop:
            # data['x'][:, :3] = 0
            # Du: modify so that it copes with intensity instead
            data['x'][:, :1] = 0
        return data


@DataTransforms.register_module()
class ChromaticPerDropGPU(object):
    def __init__(self, color_drop=0.2, **kwargs):
        self.color_drop = color_drop

    def __call__(self, data):
        colors_drop = (torch.rand((data['x'].shape[0], 1)) > self.color_drop).to(torch.float32)
        # data['x'][:, :3] *= colors_drop
        # Du: modify so that it copes with intensity instead
        data['x'][:, :1] *= colors_drop
        return data


@DataTransforms.register_module()
class ChromaticNormalize(object):
    def __init__(self,
                 color_mean=[0.5136457, 0.49523646, 0.44921124],
                 color_std=[0.18308958, 0.18415008, 0.19252081],
                 **kwargs):
        self.color_mean = torch.from_numpy(np.array(color_mean)).to(torch.float32)
        self.color_std = torch.from_numpy(np.array(color_std)).to(torch.float32)

    def __call__(self, data):
        device = data['x'].device
        if data['x'][:, :3].max() > 1:
            data['x'][:, :3] /= 255.
        data['x'][:, :3] = (data['x'][:, :3] - self.color_mean.to(device)) / self.color_std.to(device)
        return data


def one_hot(x, num_classes, on_value=1., off_value=0., device='cuda'):
    x = x.long().view(-1, 1)
    return torch.full((x.size()[0], num_classes), off_value, device=device).scatter_(1, x, on_value)


def mixup_target(target, num_classes, lam=1., smoothing=0.0, device='cuda'):
    off_value = smoothing / num_classes
    on_value = 1. - smoothing + off_value
    y1 = one_hot(target, num_classes, on_value=on_value, off_value=off_value, device=device)
    y2 = one_hot(target.flip(0), num_classes, on_value=on_value, off_value=off_value, device=device)
    return y1 * lam + y2 * (1. - lam)


class Cutmix:
    """ Cutmix that applies different params to each element or whole batch
    Update: 1. random cutmix does not work on classification (ScanObjectNN, PointNext), April 7, 2022
    Args:
        cutmix_alpha (float): cutmix alpha value, cutmix is active if > 0.
        prob (float): probability of applying mixup or cutmix per batch or element
        label_smoothing (float): apply label smoothing to the mixed target tensor
        num_classes (int): number of classes for target
    """

    def __init__(self, cutmix_alpha=0.3, prob=1.0,
                 label_smoothing=0.1, num_classes=1000):
        self.cutmix_alpha = cutmix_alpha
        self.mix_prob = prob
        self.label_smoothing = label_smoothing
        self.num_classes = num_classes

    def _mix_batch(self, data):
        lam = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
        # the trianing batches should have same size. 
        if hasattr(data, 'keys'):  # data is a dict
            # pos, feat? 
            N = data['pos'].shape[1]
            n_mix = int(N * lam)
            data['pos'][:, -n_mix:] = data['pos'].flip(0)[:, -n_mix:]

            if 'x' in data.keys():
                data['x'][:, :, -n_mix:] = data['x'].flip(0)[:, :, -n_mix:]
        else:
            data[:, -n_mix:] = data.flip(0)[:, -n_mix:]
        return lam

    def __call__(self, data, target):
        device = data['pos'].device if hasattr(data, 'keys') else data.device
        lam = self._mix_batch(data)
        target = mixup_target(target, self.num_classes, lam, self.label_smoothing, device)
        return data, target