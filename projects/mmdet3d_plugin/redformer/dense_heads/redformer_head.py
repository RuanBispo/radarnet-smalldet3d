# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------
from __future__ import annotations

import copy

import cv2 as cv
import os
import matplotlib.pyplot as plt
import mmcv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Linear, bias_init_with_prob
from mmcv.cnn.bricks.transformer import build_positional_encoding
from mmcv.ops.multi_scale_deform_attn import MultiScaleDeformableAttention
from mmcv.runner import auto_fp16, force_fp32
from mmcv.utils import TORCH_VERSION, digit_version
from mmdet3d.core.bbox.coders import build_bbox_coder
from mmdet.core import multi_apply, reduce_mean
from mmdet.models import HEADS
from mmdet.models.dense_heads import DETRHead
from mmdet.models.utils.transformer import inverse_sigmoid
from mmdet3d.core.bbox import LiDARInstance3DBoxes

from projects.mmdet3d_plugin.core.bbox.util import normalize_bbox, denormalize_bbox
from projects.mmdet3d_plugin.models.utils.bricks import run_time
from projects.mmdet3d_plugin.models.utils.visual import save_tensor
from projects.mmdet3d_plugin.redformer.modules.encoder import BEVFormerEncoder


class RadarBackbone(nn.Module):
    """Radar backbone neural network module designed for radar data.

    Attributes:
    """

    __slots__ = ("max_num_radar_per_grid", "embedding_dim", "_embedding", "_gated_unit", "_mha")

    def __init__(self, max_num_radar_per_grid: int, hidden_size: int) -> None:
        super().__init__()
        self.max_num_radar_per_grid = max_num_radar_per_grid
        self.embedding_dim = int(max_num_radar_per_grid // 2)

        # embedding layer for number encoding
        self._embedding = nn.Embedding(
            num_embeddings=self.max_num_radar_per_grid, embedding_dim=self.embedding_dim
        )
        # gated linear unit to filter noises
        self._gated_unit = nn.ModuleDict(
            {
                "filter_linear": nn.Sequential(
                    nn.Linear(self.embedding_dim*10, hidden_size), nn.Sigmoid()
                ),
                "signal_linear": nn.Sequential(
                    nn.Linear(self.embedding_dim*10, hidden_size), nn.Tanh()
                ),
            }
        )
        # multi-head attention
        # self._mha = nn.MultiheadAttention(
        #     embed_dim=hidden_size,
        #     num_heads=4,
        #     dropout=0.0,
        #     batch_first=True
        # )

        # self.defomable_selfatten = MultiScaleDeformableAttention(
        #     embed_dims=hidden_size,
        #     num_heads=8,
        #     num_levels=1,
        #     batch_first=True
        # )

    def forward(self, input: torch.Tensor) -> torch.FloatTensor:
        input = input.long()
        # _, width, length = input.shape
        _, width, length, c = input.shape #[1, 150, 150, 8]
        # print('c=', input.shape)
        # input = input.view(-1)
        input = input.view(width*length, -1)

        # embedding forward
        hidden = self._embedding.forward(input)
        hidden = hidden.view(width*length, self.embedding_dim*c)
        assert hidden.shape == (width * length, c * self.embedding_dim) # (16384, 5)

        # gated unit forward
        filter_output = self._gated_unit["filter_linear"].forward(hidden)
        signal_output = self._gated_unit["signal_linear"].forward(hidden)
        hidden = filter_output * signal_output

        output = hidden

        return output


@HEADS.register_module()
class REDFormerHead(DETRHead):
    """Head of Detr3D.

    Args:
        with_box_refine (bool): Whether to refine the reference points
            in the decoder. Defaults to False.
        as_two_stage (bool) : Whether to generate the proposal from
            the outputs of encoder.
        transformer (obj:`ConfigDict`): ConfigDict is used for building
            the Encoder and Decoder.
        bev_h, bev_w (int): spatial shape of BEV queries.
    """

    def __init__(
        self,
        *args,
        with_box_refine=False,
        as_two_stage=False,
        transformer=None,
        bbox_coder=None,
        num_cls_fcs=2,
        code_weights=None,
        bev_h=30,
        bev_w=30,
        pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],  
        **kwargs,
    ):
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.fp16_enabled = False
        self.xmin, self.xmax, self.ymin, self.ymax = (
            pc_range[0],
            pc_range[3],
            pc_range[1],
            pc_range[4],
        ) 

        self.with_box_refine = with_box_refine
        self.as_two_stage = as_two_stage
        if self.as_two_stage:
            transformer["as_two_stage"] = self.as_two_stage
        if "code_size" in kwargs:
            self.code_size = kwargs["code_size"]
        else:
            self.code_size = 10
        if code_weights is not None:
            self.code_weights = code_weights
        else:
            self.code_weights = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.pc_range = self.bbox_coder.pc_range
        self.real_w = self.pc_range[3] - self.pc_range[0]
        self.real_h = self.pc_range[4] - self.pc_range[1]
        self.num_cls_fcs = num_cls_fcs - 1
        super().__init__(*args, transformer=transformer, **kwargs)
        self.code_weights = nn.Parameter(
            torch.tensor(self.code_weights, requires_grad=False), requires_grad=False
        )

        self.weather_head = nn.Linear(self.embed_dims, 2)
        self.tod_head = nn.Linear(self.embed_dims, 2)  # tod: time_of_day
        # self.sec_regression= nn.Linear(self.code_size + 6, self.code_size)
        # self.radar_backbone = torch.nn.Conv2d(in_channels=1, out_channels=256, kernel_size=1, stride=1, padding=0)

        # embedding backbone
        # self.radar_emb = torch.nn.Sequential(
        #     torch.nn.Embedding(num_embeddings=10, embedding_dim=5),
        #     torch.nn.Linear(5, 256, bias=False)
        # )
        self.radar_emb = RadarBackbone(max_num_radar_per_grid=50, hidden_size=256)

    def forward_context(self, bev_embed):
        """

        Args:
            bev_embed: (bev_h * bev_w) x 1 x embed_dims
        """
        assert bev_embed.shape[0] == self.bev_w * self.bev_h
        x = torch.mean(bev_embed, dim=0)
        x = torch.flatten(x)
        return self.weather_head(x), self.tod_head(x)

    def _init_layers(self):
        """Initialize classification branch and regression branch of head."""
        cls_branch = []
        for _ in range(self.num_reg_fcs):
            cls_branch.append(Linear(self.embed_dims, self.embed_dims))
            cls_branch.append(nn.LayerNorm(self.embed_dims))
            cls_branch.append(nn.ReLU(inplace=True))
        cls_branch.append(Linear(self.embed_dims, self.cls_out_channels))
        fc_cls = nn.Sequential(*cls_branch)

        reg_branch = []
        for _ in range(self.num_reg_fcs):
            reg_branch.append(Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU())
        reg_branch.append(Linear(self.embed_dims, self.code_size))
        reg_branch = nn.Sequential(*reg_branch)
        
        # self.sec_reg = nn.Sequential(
        #     nn.Linear(self.code_size + 15, self.code_size + 15), 
        #     nn.ReLU(),
        #     nn.Linear(self.code_size + 15, self.code_size + 15), 
        #     nn.ReLU(),
        #     Linear(self.code_size + 15, 5)
        # )

        def _get_clones(module, N):
            return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

        # last reg_branch is used to generate proposal from
        # encode feature map when as_two_stage is True.
        num_pred = (
            (self.transformer.decoder.num_layers + 1)
            if self.as_two_stage
            else self.transformer.decoder.num_layers
        )

        if self.with_box_refine:
            self.cls_branches = _get_clones(fc_cls, num_pred)
            self.reg_branches = _get_clones(reg_branch, num_pred)
        else:
            self.cls_branches = nn.ModuleList([fc_cls for _ in range(num_pred)])
            self.reg_branches = nn.ModuleList([reg_branch for _ in range(num_pred)])

        if not self.as_two_stage:
            self.bev_embedding = nn.Embedding(self.bev_h * self.bev_w, self.embed_dims)
            self.query_embedding = nn.Embedding(self.num_query, self.embed_dims * 2)

    def init_weights(self):
        """Initialize weights of the DeformDETR head."""
        self.transformer.init_weights()
        if self.loss_cls.use_sigmoid:
            bias_init = bias_init_with_prob(0.01)
            for m in self.cls_branches:
                nn.init.constant_(m[-1].bias, bias_init)

    @auto_fp16(apply_to="mlvl_feats")
    def forward(self, mlvl_feats, img_metas, prev_bev=None, only_bev=False, radar_hist=None, radar_points=None, gt_bboxes_3d=None):
        """Forward function.

        Args:
            mlvl_feats (tuple[Tensor]): Features from the upstream
                network, each is a 5D-tensor with shape
                (B, N, C, H, W).
            prev_bev: previous bev features
            only_bev: only compute BEV features with encoder.
        Returns:
            all_cls_scores (Tensor): Outputs from the classification head, \
                shape [nb_dec, bs, num_query, cls_out_channels]. Note \
                cls_out_channels should include background.
            all_bbox_preds (Tensor): Sigmoid outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, theta, vx, vy). \
                Shape [nb_dec, bs, num_query, 9].
        """

        bs, num_cam, _, _, _ = mlvl_feats[0].shape
        dtype = mlvl_feats[0].dtype
        object_query_embeds = self.query_embedding.weight.to(dtype)
        bev_queries = self.bev_embedding.weight.to(dtype)
        embed_dim = bev_queries.shape[1]

        if radar_hist is not None:
            radar_hist = radar_hist.long()  
            radar_bev = self.radar_emb(radar_hist) #.view(-1, 256)
            bev_queries = bev_queries + radar_bev

        bev_mask = torch.zeros((bs, self.bev_h, self.bev_w), device=bev_queries.device).to(dtype)
        bev_pos = self.positional_encoding(bev_mask).to(dtype)

        if only_bev:  # only use encoder to obtain BEV features, TODO: refine the workaround
            return self.transformer.get_bev_features(
                mlvl_feats,
                bev_queries,
                self.bev_h,
                self.bev_w,
                grid_length=(self.real_h / self.bev_h, self.real_w / self.bev_w),
                bev_pos=bev_pos,
                img_metas=img_metas,
                prev_bev=prev_bev,
            )
        else:
            outputs = self.transformer(
                mlvl_feats,
                bev_queries,
                object_query_embeds,
                self.bev_h,
                self.bev_w,
                # radar_bev,
                grid_length=(self.real_h / self.bev_h, self.real_w / self.bev_w),
                bev_pos=bev_pos,
                reg_branches=self.reg_branches if self.with_box_refine else None,  # noqa:E501
                cls_branches=self.cls_branches if self.as_two_stage else None,
                img_metas=img_metas,
                prev_bev=prev_bev
            )

        bev_embed, hs, init_reference, inter_references = outputs
        hs = hs.permute(0, 2, 1, 3)
        
        outputs_classes = []
        outputs_coords = []
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.cls_branches[lvl](hs[lvl])
            tmp = self.reg_branches[lvl](hs[lvl])
            

            # TODO: check the shape of reference
            assert reference.shape[-1] == 3
            tmp[..., 0:2] += reference[..., 0:2]
            tmp[..., 0:2] = tmp[..., 0:2].sigmoid()
            tmp[..., 4:5] += reference[..., 2:3]
            tmp[..., 4:5] = tmp[..., 4:5].sigmoid()
            tmp[..., 0:1] = (
                tmp[..., 0:1] * (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0]
            )
            tmp[..., 1:2] = (
                tmp[..., 1:2] * (self.pc_range[4] - self.pc_range[1]) + self.pc_range[1]
            )
            tmp[..., 4:5] = (
                tmp[..., 4:5] * (self.pc_range[5] - self.pc_range[2]) + self.pc_range[2]
            )

            # TODO: check if using sigmoid
            outputs_coord = tmp
            
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)

        outputs_classes = torch.stack(outputs_classes)
        outputs_coords = torch.stack(outputs_coords)
        outputs_weather, outputs_tod = self.forward_context(bev_embed)

        outs = {
            "bev_embed": bev_embed,
            "all_cls_scores": outputs_classes,
            "all_bbox_preds": outputs_coords,
            "enc_cls_scores": None,
            "enc_bbox_preds": None,
            "weather_preds": outputs_weather,
            "tod_preds": outputs_tod,
        }

        return outs

    def _get_target_single(
        self, cls_score, bbox_pred, gt_labels, gt_bboxes, gt_bboxes_ignore=None
    ):
        """ "Compute regression and classification targets for one image.
        Outputs from a single decoder layer of a single feature level are used.
        Args:
            cls_score (Tensor): Box score logits from a single decoder layer
                for one image. Shape [num_query, cls_out_channels].
            bbox_pred (Tensor): Sigmoid outputs from a single decoder layer
                for one image, with normalized coordinate (cx, cy, w, h) and
                shape [num_query, 4].
            gt_bboxes (Tensor): Ground truth bboxes for one image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (Tensor): Ground truth class indices for one image
                with shape (num_gts, ).
            gt_bboxes_ignore (Tensor, optional): Bounding boxes
                which can be ignored. Default None.
        Returns:
            tuple[Tensor]: a tuple containing the following for one image.
                - labels (Tensor): Labels of each image.
                - label_weights (Tensor]): Label weights of each image.
                - bbox_targets (Tensor): BBox targets of each image.
                - bbox_weights (Tensor): BBox weights of each image.
                - pos_inds (Tensor): Sampled positive indices for each image.
                - neg_inds (Tensor): Sampled negative indices for each image.
        """

        num_bboxes = bbox_pred.size(0)
        # assigner and sampler
        gt_c = gt_bboxes.shape[-1]

        assign_result = self.assigner.assign(
            bbox_pred, cls_score, gt_bboxes, gt_labels, gt_bboxes_ignore
        )

        sampling_result = self.sampler.sample(assign_result, bbox_pred, gt_bboxes)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds

        # label targets
        labels = gt_bboxes.new_full((num_bboxes,), self.num_classes, dtype=torch.long)
        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        label_weights = gt_bboxes.new_ones(num_bboxes)

        # bbox targets
        bbox_targets = torch.zeros_like(bbox_pred)[..., :gt_c]
        bbox_weights = torch.zeros_like(bbox_pred)
        bbox_weights[pos_inds] = 1.0

        # DETR
        bbox_targets[pos_inds] = sampling_result.pos_gt_bboxes
        return (labels, label_weights, bbox_targets, bbox_weights, pos_inds, neg_inds)

    def get_targets(
        self,
        cls_scores_list,
        bbox_preds_list,
        gt_bboxes_list,
        gt_labels_list,
        gt_bboxes_ignore_list=None,
    ):
        """"Compute regression and classification targets for a batch image.
        Outputs from a single decoder layer of a single feature level are used.
        Args:
            cls_scores_list (list[Tensor]): Box score logits from a single
                decoder layer for each image with shape [num_query,
                cls_out_channels].
            bbox_preds_list (list[Tensor]): Sigmoid outputs from a single
                decoder layer for each image, with normalized coordinate
                (cx, cy, w, h) and shape [num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.
        Returns:
            tuple: a tuple containing the following targets.
                - labels_list (list[Tensor]): Labels for all images.
                - label_weights_list (list[Tensor]): Label weights for all \
                    images.
                - bbox_targets_list (list[Tensor]): BBox targets for all \
                    images.
                - bbox_weights_list (list[Tensor]): BBox weights for all \
                    images.
                - num_total_pos (int): Number of positive samples in all \
                    images.
                - num_total_neg (int): Number of negative samples in all \
                    images.
        """
        assert gt_bboxes_ignore_list is None, "Only supports for gt_bboxes_ignore setting to None."
        num_imgs = len(cls_scores_list)
        gt_bboxes_ignore_list = [gt_bboxes_ignore_list for _ in range(num_imgs)]

        (
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            pos_inds_list,
            neg_inds_list,
        ) = multi_apply(
            self._get_target_single,
            cls_scores_list,
            bbox_preds_list,
            gt_labels_list,
            gt_bboxes_list,
            gt_bboxes_ignore_list,
        )
        num_total_pos = sum(inds.numel() for inds in pos_inds_list)
        num_total_neg = sum(inds.numel() for inds in neg_inds_list)
        return (
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            num_total_pos,
            num_total_neg,
        )

    def loss_single(
        self, cls_scores, bbox_preds, gt_bboxes_list, gt_labels_list, gt_bboxes_ignore_list=None
    ):
        """"Loss function for outputs from a single decoder layer of a single feature level.

        Args:
            cls_scores (Tensor): Box score logits from a single decoder layer
                for all images. Shape [bs, num_query, cls_out_channels].
            bbox_preds (Tensor): Sigmoid outputs from a single decoder layer
                for all images, with normalized coordinate (cx, cy, w, h) and
                shape [bs, num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.
        Returns:
            dict[str, Tensor]: A dictionary of loss components for outputs from
                a single decoder layer.
        """
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]
        cls_reg_targets = self.get_targets(
            cls_scores_list, bbox_preds_list, gt_bboxes_list, gt_labels_list, gt_bboxes_ignore_list
        )
        (
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            num_total_pos,
            num_total_neg,
        ) = cls_reg_targets
        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)

        # classification loss
        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(cls_scores.new_tensor([cls_avg_factor]))

        cls_avg_factor = max(cls_avg_factor, 1)
        loss_cls = self.loss_cls(cls_scores, labels, label_weights, avg_factor=cls_avg_factor)

        # Compute the average number of gt boxes across all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # regression L1 loss
        bbox_preds = bbox_preds.reshape(-1, bbox_preds.size(-1))
        normalized_bbox_targets = normalize_bbox(bbox_targets, self.pc_range)
        isnotnan = torch.isfinite(normalized_bbox_targets).all(dim=-1)
        bbox_weights = bbox_weights * self.code_weights

        loss_bbox = self.loss_bbox(
            bbox_preds[isnotnan, :10],
            normalized_bbox_targets[isnotnan, :10],
            bbox_weights[isnotnan, :10],
            avg_factor=num_total_pos,
        )
        if digit_version(TORCH_VERSION) >= digit_version("1.8"):
            loss_cls = torch.nan_to_num(loss_cls)
            loss_bbox = torch.nan_to_num(loss_bbox)
        return loss_cls, loss_bbox

    @force_fp32(apply_to=("preds_dicts"))
    def loss(
        self,
        gt_bboxes_list,
        gt_labels_list,
        preds_dicts,
        gt_bboxes_ignore=None,
        img_metas=None,
        gt_rain=None,
        gt_tod=None,
    ):
        """ "Loss function.
        Args:
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            preds_dicts:
                all_cls_scores (Tensor): Classification score of all
                    decoder layers, has shape
                    [nb_dec, bs, num_query, cls_out_channels].
                all_bbox_preds (Tensor): Sigmoid regression
                    outputs of all decode layers. Each is a 4D-tensor with
                    normalized coordinate format (cx, cy, w, h) and shape
                    [nb_dec, bs, num_query, 4].
                enc_cls_scores (Tensor): Classification scores of
                    points on encode feature map , has shape
                    (N, h*w, num_classes). Only be passed when as_two_stage is
                    True, otherwise is None.
                enc_bbox_preds (Tensor): Regression results of each points
                    on the encode feature map, has shape (N, h*w, 4). Only be
                    passed when as_two_stage is True, otherwise is None.
            gt_bboxes_ignore (list[Tensor], optional): Bounding boxes
                which can be ignored for each image. Default None.
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert gt_bboxes_ignore is None, (
            f"{self.__class__.__name__} only supports " f"for gt_bboxes_ignore setting to None."
        )

        all_cls_scores = preds_dicts["all_cls_scores"]
        all_bbox_preds = preds_dicts["all_bbox_preds"]
        enc_cls_scores = preds_dicts["enc_cls_scores"]
        enc_bbox_preds = preds_dicts["enc_bbox_preds"]

        num_dec_layers = len(all_cls_scores)
        device = gt_labels_list[0].device

        gt_bboxes_list = [
            torch.cat((gt_bboxes.gravity_center, gt_bboxes.tensor[:, 3:]), dim=1).to(device)
            for gt_bboxes in gt_bboxes_list
        ]

        all_gt_bboxes_list = [gt_bboxes_list for _ in range(num_dec_layers)]
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
        all_gt_bboxes_ignore_list = [gt_bboxes_ignore for _ in range(num_dec_layers)]

        losses_cls, losses_bbox = multi_apply(
            self.loss_single,
            all_cls_scores,
            all_bbox_preds,
            all_gt_bboxes_list,
            all_gt_labels_list,
            all_gt_bboxes_ignore_list,
        )

        ### Mayson
        weather_preds = preds_dicts["weather_preds"]
        if len(weather_preds.shape) == 1:
            weather_preds = torch.unsqueeze(weather_preds, 0)
        tod_preds = preds_dicts["tod_preds"]
        if len(tod_preds.shape) == 1:
            tod_preds = torch.unsqueeze(tod_preds, 0)
        ###

        loss_dict = dict()
        # loss of proposal generated from encode feature map.
        if enc_cls_scores is not None:
            binary_labels_list = [
                torch.zeros_like(gt_labels_list[i]) for i in range(len(all_gt_labels_list))
            ]
            enc_loss_cls, enc_losses_bbox = self.loss_single(
                enc_cls_scores,
                enc_bbox_preds,
                gt_bboxes_list,
                binary_labels_list,
                gt_bboxes_ignore,
            )
            loss_dict["enc_loss_cls"] = enc_loss_cls
            loss_dict["enc_loss_bbox"] = enc_losses_bbox

        # loss from the last decoder layer
        loss_dict["loss_cls"] = losses_cls[-1]
        loss_dict["loss_bbox"] = losses_bbox[-1]

        # loss from other decoder layers
        num_dec_layer = 0
        for loss_cls_i, loss_bbox_i in zip(losses_cls[:-1], losses_bbox[:-1]):
            loss_dict[f"d{num_dec_layer}.loss_cls"] = loss_cls_i
            loss_dict[f"d{num_dec_layer}.loss_bbox"] = loss_bbox_i
            num_dec_layer += 1

        # Mayson: loss from context
        if gt_rain is None and gt_tod is None:
            pass
        else:
            loss_context = 0.0
            if gt_rain is not None:
                loss_context += F.cross_entropy(weather_preds, gt_rain)
            if gt_tod is not None:
                loss_context += F.cross_entropy(tod_preds, gt_tod)
            loss_dict["loss_context"] = loss_context

        return loss_dict

    @force_fp32(apply_to=("preds_dicts"))
    def get_bboxes(self, preds_dicts, img_metas, rescale=False):
        """Generate bboxes from bbox head predictions.

        Args:
            preds_dicts (tuple[list[dict]]): Prediction results.
            img_metas (list[dict]): Point cloud and image's meta info.
        Returns:
            list[dict]: Decoded bbox, scores and labels after nms.
        """

        preds_dicts = self.bbox_coder.decode(preds_dicts)

        num_samples = len(preds_dicts)
        ret_list = []
        for i in range(num_samples):
            preds = preds_dicts[i]
            bboxes = preds["bboxes"]

            bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5] * 0.5

            code_size = bboxes.shape[-1]
            bboxes = img_metas[i]["box_type_3d"](bboxes, code_size)
            scores = preds["scores"]
            labels = preds["labels"]

            ret_list.append([bboxes, scores, labels])

        return ret_list
