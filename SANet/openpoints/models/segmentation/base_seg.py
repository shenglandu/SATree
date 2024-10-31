"""
Author: PointNeXt
"""
import copy
from typing import List
import torch
import torch.nn as nn
import logging
from ..build import MODELS, build_model_from_cfg
from ..layers import create_linearblock, create_convblock1d
import torch.nn.functional as F


@MODELS.register_module()
class BaseSeg(nn.Module):
    def __init__(self,
                 encoder_args=None,
                 decoder_args=None,
                 cls_args=None,
                 longrange_args=None,
                 group_args={'NAME': 'ballquery', 'radius': 0.5},
                 **kwargs):
        super().__init__()
        self.encoder = build_model_from_cfg(encoder_args)
        if decoder_args is not None:
            decoder_args_merged_with_encoder = copy.deepcopy(encoder_args)
            decoder_args_merged_with_encoder.update(decoder_args)
            decoder_args_merged_with_encoder.encoder_channel_list = self.encoder.channel_list if hasattr(self.encoder,
                                                                                                         'channel_list') else None
            self.decoder = build_model_from_cfg(decoder_args_merged_with_encoder)
            self.score_decoder = build_model_from_cfg(decoder_args_merged_with_encoder)
            self.offset_decoder = build_model_from_cfg(decoder_args_merged_with_encoder)

        else:
            self.decoder = None

        if longrange_args is not None:
            self.decoder.longrange = build_model_from_cfg(longrange_args)
        else:
            self.longrange = None

        if cls_args is not None:
            if hasattr(self.decoder, 'out_channels'):
                in_channels = self.decoder.out_channels
            elif hasattr(self.encoder, 'out_channels'):
                in_channels = self.encoder.out_channels
            else:
                in_channels = cls_args.get('in_channels', None)
            cls_args.in_channels = in_channels
            self.head = build_model_from_cfg(cls_args)

            # Du: append a branch to predict scores
            score_head = []
            score_head.append(create_convblock1d(in_channels, in_channels, norm_args={'norm': 'bn1d'}, act_args={'act': 'relu'}))
            score_head.append(nn.Dropout(0.5))
            score_head.append(create_convblock1d(in_channels, 1, act_args={'act': 'relu'}))
            self.score_head = nn.Sequential(*score_head)

            # Du: append another branch to predict offset
            offset_head = []
            offset_head.append(create_convblock1d(in_channels, in_channels, norm_args={'norm': 'bn1d'}, act_args={'act': 'relu'}))
            offset_head.append(create_convblock1d(in_channels, 3, act_args=None))
            self.offset_head = nn.Sequential(*offset_head)

        else:
            self.head = None

        self.ignore_label = -100

    def forward(self, data):
        p, f = self.encoder.forward_seg_feat(data)
        # if self.decoder is not None:
        #     f = self.decoder(p, f).squeeze(-1)
        # if self.head is not None:
        #     f = self.head(f)
        # return f
        # Du: modify the output
        # f2 = f.copy()
        if self.decoder is not None:
            f1 = self.decoder(p, f).squeeze(-1)
            f2 = self.score_decoder(p, f).squeeze(-1)
            f3 = self.offset_decoder(p, f).squeeze(-1)
        if self.head is not None:
            f1 = self.head(f1)
            f2 = self.score_head(f2)
            f3 = self.offset_head(f3)
            return f1, f2, f3
            # return f1, f3
        else:
            return f

    def score_loss(self, pred, target):
        pred = pred.squeeze(1)
        # if ignore_idx is not None:
        #     valid_mask = torch.where(semantic_labels == ignore_idx, 0.0, 1.0).contiguous()
        #     pred = pred * valid_mask
        #     target = target * valid_mask

        loss = F.mse_loss(pred, target)
        return loss

    def offset_loss(self, offsets, offset_labels, instance_labels):
        offsets = offsets.permute(0, 2, 1).reshape(-1, 3)
        offset_labels = offset_labels.reshape(-1, 3)
        # normalize offset labels
        # offset_labels = F.normalize(offset_labels, dim=-1)
        instance_labels = instance_labels.flatten()

        valid_idx = instance_labels != self.ignore_label

        if valid_idx.sum() == 0:
            offset_loss = 0 * offsets.sum()
        else:
            # offset_loss = F.l1_loss(offsets[valid_idx], offset_labels[valid_idx], reduction='sum') / valid_idx.sum()
            offset_loss = F.mse_loss(offsets[valid_idx], offset_labels[valid_idx], reduction='sum') / valid_idx.sum()

        return offset_loss


@MODELS.register_module()
class BasePartSeg(BaseSeg):
    def __init__(self, encoder_args=None, decoder_args=None, cls_args=None, **kwargs):
        super().__init__(encoder_args, decoder_args, cls_args, **kwargs)

    def forward(self, p0, f0=None, cls0=None):
        if hasattr(p0, 'keys'):
            p0, f0, cls0 = p0['pos'], p0['x'], p0['cls']
        else:
            if f0 is None:
                f0 = p0.transpose(1, 2).contiguous()
        p, f = self.encoder.forward_seg_feat(p0, f0)
        if self.decoder is not None:
            f = self.decoder(p, f, cls0).squeeze(-1)
        elif isinstance(f, list):
            f = f[-1]
        if self.head is not None:
            f = self.head(f)
        return f


@MODELS.register_module()
class VariableSeg(BaseSeg):
    def __init__(self,
                 encoder_args=None,
                 decoder_args=None,
                 cls_args=None,
                 **kwargs):
        super().__init__(encoder_args, decoder_args, cls_args)
        if kwargs:
            logging.warning(f"kwargs: {kwargs} are not used in {__class__.__name__}")

    def forward(self, data):
        p, f, b = self.encoder.forward_seg_feat(data)
        f = self.decoder(p, f, b).squeeze(-1)
        return self.head(f)


@MODELS.register_module()
class SegHead(nn.Module):
    def __init__(self,
                 num_classes, in_channels,
                 mlps=None,
                 norm_args={'norm': 'bn1d'},
                 act_args={'act': 'relu'},
                 dropout=0.5,
                 globals=None, 
                 **kwargs
                 ):
        """A scene segmentation head for ResNet backbone.
        Args:
            num_classes: class num.
            in_channles: the base channel num.
            globals: global features to concat. [max,avg]. Set to None if do not concat any.
        Returns:
            logits: (B, num_classes, N)
        """
        super().__init__()
        if kwargs:
            logging.warning(f"kwargs: {kwargs} are not used in {__class__.__name__}")
        if globals is not None:
            self.globals = globals.split(',')
            multiplier = len(self.globals) + 1
        else:
            self.globals = None
            multiplier = 1
        in_channels *= multiplier
        
        if mlps is None:
            mlps = [in_channels, in_channels] + [num_classes]
        else:
            if not isinstance(mlps, List):
                mlps = [mlps]
            mlps = [in_channels] + mlps + [num_classes]
        heads = []
        for i in range(len(mlps) - 2):
            heads.append(create_convblock1d(mlps[i], mlps[i + 1],
                                            norm_args=norm_args,
                                            act_args=act_args))
            if dropout:
                heads.append(nn.Dropout(dropout))

        heads.append(create_convblock1d(mlps[-2], mlps[-1], act_args=None))
        self.head = nn.Sequential(*heads)

    def forward(self, end_points):
        if self.globals is not None: 
            global_feats = [] 
            for feat_type in self.globals:
                if 'max' in feat_type:
                    global_feats.append(torch.max(end_points, dim=-1, keepdim=True)[0])
                elif feat_type in ['avg', 'mean']:
                    global_feats.append(torch.mean(end_points, dim=-1, keepdim=True))
            global_feats = torch.cat(global_feats, dim=1).expand(-1, -1, end_points.shape[-1])
            end_points = torch.cat((end_points, global_feats), dim=1)
        logits = self.head(end_points)
        return logits


@MODELS.register_module()
class VariableSegHead(nn.Module):
    def __init__(self,
                 num_classes, in_channels,
                 norm_args={'norm': 'bn1d'},
                 act_args={'act': 'relu'},
                 dropout=0.5,
                 **kwargs
                 ):
        """A scene segmentation head for ResNet backbone.
        Args:
            num_classes: class num.
            in_channles: the base channel num.
        Returns:
            logits: (B, num_classes, N)
        """
        super().__init__()
        if kwargs:
            logging.warning(f"kwargs: {kwargs} are not used in {__class__.__name__}")
        mlps = [in_channels, in_channels] + [num_classes]

        heads = []
        print(mlps, norm_args, act_args)
        for i in range(len(mlps) - 2):
            heads.append(create_linearblock(mlps[i], mlps[i + 1],
                                            norm_args=norm_args,
                                            act_args=act_args))
            if dropout:
                heads.append(nn.Dropout(dropout))

        heads.append(create_linearblock(mlps[-2], mlps[-1], act_args=None))
        self.head = nn.Sequential(*heads)

    def forward(self, end_points):
        logits = self.head(end_points)
        return logits

@MODELS.register_module()
class MultiSegHead(nn.Module):
    def __init__(self,
                 num_classes, in_channels,
                 norm_args={'norm': 'bn1d'},
                 act_args={'act': 'relu'},
                 dropout=0,
                 shape_classes=16,
                 num_parts=[4, 2, 2, 4, 4, 3, 3, 2, 4, 2, 6, 2, 3, 3, 3, 3],
                 **kwargs
                 ):
        """A scene segmentation head for ResNet backbone.
        Args:
            num_classes: class num.
            in_channles: the base channel num.
        Returns:
            logits: (B, num_classes, N)
        """
        super().__init__()
        if kwargs:
            logging.warning(f"kwargs: {kwargs} are not used in {__class__.__name__}")
        mlps = [in_channels, in_channels] + [shape_classes]
        self.multi_shape_heads = []

        self.num_parts=num_parts
        print(mlps, norm_args, act_args)
        self.shape_classes = shape_classes
        self.multi_shape_heads = nn.ModuleList()
        for i in range(shape_classes):
            head=[]
            for j in range(len(mlps) - 2):

                head.append(create_convblock1d(mlps[j], mlps[j + 1],
                                                norm_args=norm_args,
                                                act_args=act_args))
                if dropout:
                    head.append(nn.Dropout(dropout))
                head.append(nn.Conv1d(mlps[-2], num_parts[i], kernel_size=1, bias=True))
            self.multi_shape_heads.append(nn.Sequential(*head))

        # heads.append(create_linearblock(mlps[-2], mlps[-1], act_args=None))

    def forward(self, end_points):
        logits_all_shapes = []
        for i in range(self.shape_classes):# per 16 shapes
            logits_all_shapes.append(self.multi_shape_heads[i](end_points))
        # logits = self.head(end_points)
        return logits_all_shapes
