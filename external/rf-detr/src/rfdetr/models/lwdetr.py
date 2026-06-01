# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Copied and modified from LW-DETR (https://github.com/Atten4Vis/LW-DETR)
# Copyright (c) 2024 Baidu. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Conditional DETR (https://github.com/Atten4Vis/ConditionalDETR)
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------

'''
здесь rf-detr получает основную temporal логику.
модель умеет отличать текущий кадр от референсных, прогоняет соседние кадры
через тот же детектор и затем уточняет query-представления текущего кадра
через temporal fusion перед головами классификации и bbox.
'''

"""
LW-DETR model and criterion classes
"""

import copy
import math
from typing import TYPE_CHECKING, Callable, Optional

import torch
from torch import nn

if TYPE_CHECKING:
    from rfdetr.config import ModelConfig, TrainConfig

from rfdetr.models._defaults import MODEL_DEFAULTS, ModelDefaults
from rfdetr.models._types import BuilderArgs
from rfdetr.models.backbone import build_backbone

# Backward-compat re-exports: loss functions that used to live in this module
from rfdetr.models.criterion import (  # noqa: F401 — backward compat
    SetCriterion,
    dice_loss,
    dice_loss_jit,
    position_supervised_loss,
    sigmoid_ce_loss,
    sigmoid_ce_loss_jit,
    sigmoid_focal_loss,
    sigmoid_varifocal_loss,
)
from rfdetr.models.heads.segmentation import SegmentationHead
from rfdetr.models.matcher import build_matcher
from rfdetr.models.math import MLP
from rfdetr.models.postprocess import PostProcess
from rfdetr.models.temporal import TemporalQueryFusion  # новый блок, который смешивает query текущего и соседних кадров
from rfdetr.models.transformer import build_transformer
from rfdetr.utilities.tensors import NestedTensor, nested_tensor_from_tensor_list


def _resize_linear(linear: nn.Linear, num_classes: int) -> nn.Linear:
    """Return a new :class:`~torch.nn.Linear` resized to *num_classes* outputs.

    Tiles the existing weight rows when *num_classes* is larger than the current
    output size, or truncates them when smaller.  The returned module has
    ``out_features == num_classes`` so that ``nn.Linear`` metadata stays
    consistent with the actual weight shape — a requirement for correct ONNX
    export and ``torch.jit.trace`` serialisation.

    Args:
        linear: Source linear layer whose weights are used as the starting point.
        num_classes: Target number of output features.

    Returns:
        A new :class:`~torch.nn.Linear` with ``in_features`` unchanged and
        ``out_features == num_classes``.
    """
    base = linear.weight.shape[0]
    num_repeats = int(math.ceil(num_classes / base))
    new_weight = linear.weight.detach().repeat(num_repeats, 1)[:num_classes]
    new_bias = linear.bias.detach().repeat(num_repeats)[:num_classes] if linear.bias is not None else None
    new_linear = nn.Linear(linear.in_features, num_classes, bias=new_bias is not None)
    # Copy resized weights/bias into the new layer while preserving requires_grad flags.
    with torch.no_grad():
        new_linear.weight.copy_(new_weight)
        if new_bias is not None and new_linear.bias is not None:
            new_linear.bias.copy_(new_bias)
    new_linear.weight.requires_grad = linear.weight.requires_grad
    if linear.bias is not None and new_linear.bias is not None:
        new_linear.bias.requires_grad = linear.bias.requires_grad
    return new_linear


class LWDETR(nn.Module):
    """This is the Group DETR v3 module that performs object detection"""

    def __init__(
        self,
        backbone,
        transformer,
        segmentation_head,
        num_classes,
        num_queries,
        aux_loss=False,
        group_detr=1,
        two_stage=False,
        lite_refpoint_refine=False,
        bbox_reparam=False,
        temporal_num_ref_frames=0,  # сколько reference кадров ожидает модель
        temporal_fusion_layers=1,  # сколько слоев temporal attention создавать
        temporal_dropout=0.0,  # dropout внутри temporal attention
    ):
        """Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         Conditional DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            group_detr: Number of groups to speed detr training. Default is 1.
            lite_refpoint_refine: TODO
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.segmentation_head = segmentation_head

        query_dim = 4
        self.refpoint_embed = nn.Embedding(num_queries * group_detr, query_dim)
        self.query_feat = nn.Embedding(num_queries * group_detr, hidden_dim)
        nn.init.constant_(self.refpoint_embed.weight.data, 0)

        self.backbone = backbone
        self.aux_loss = aux_loss
        self.group_detr = group_detr

        # iter update
        self.lite_refpoint_refine = lite_refpoint_refine
        if not self.lite_refpoint_refine:
            self.transformer.decoder.bbox_embed = self.bbox_embed
        else:
            self.transformer.decoder.bbox_embed = None

        self.bbox_reparam = bbox_reparam
        self.temporal_num_ref_frames = int(temporal_num_ref_frames)  # приводим к int для расчета числа каналов
        self.temporal_enabled = self.temporal_num_ref_frames > 0  # флаг отключает temporal путь для baseline
        self.temporal_query_fusion = (
            TemporalQueryFusion(  # создаем модуль, который уточняет query по соседним кадрам
                d_model=hidden_dim,  # размерность query должна совпадать с decoder hidden dim
                nhead=transformer.decoder.layers[0].self_attn.num_heads,  # берем число голов как у decoder attention
                dim_feedforward=transformer.decoder.layers[0].linear1.out_features,  # ffn размер берем из decoder слоя
                dropout=temporal_dropout,  # та же регуляризация для temporal attention
                num_layers=temporal_fusion_layers,  # количество последовательных fusion слоев
                queries_per_group=num_queries,  # нужно для group-detr, чтобы группы не мешались между собой
            )
            if self.temporal_enabled
            else None  # если temporal выключен, модель остается обычным rf-detr
        )

        # init prior_prob setting for focal loss
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value

        # init bbox_mebed
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)

        # two_stage
        self.two_stage = two_stage
        if self.two_stage:
            self.transformer.enc_out_bbox_embed = nn.ModuleList(
                [copy.deepcopy(self.bbox_embed) for _ in range(group_detr)]
            )
            self.transformer.enc_out_class_embed = nn.ModuleList(
                [copy.deepcopy(self.class_embed) for _ in range(group_detr)]
            )

        self._export = False

    def reinitialize_detection_head(self, num_classes: int) -> None:
        """Resize the detection classification head to *num_classes* outputs.

        Replaces ``self.class_embed`` (and each ``enc_out_class_embed`` when the
        model uses two-stage detection) with a new :class:`torch.nn.Linear` whose
        ``out_features`` equals *num_classes*.  When *num_classes* is larger than
        the current head the existing weights are tiled; when smaller they are
        truncated.  Replacing the module (rather than mutating ``.data``) keeps
        ``nn.Linear.out_features`` consistent with the actual weight shape, which
        is required for correct ONNX export.

        Args:
            num_classes: Target number of output classes (including background).
        """
        self.class_embed = _resize_linear(self.class_embed, num_classes)

        if self.two_stage:
            self.transformer.enc_out_class_embed = nn.ModuleList(
                [_resize_linear(m, num_classes) for m in self.transformer.enc_out_class_embed]
            )

    def export(self):
        self._export = True
        self._forward_origin = self.forward
        self.forward = self.forward_export
        for name, m in self.named_modules():
            if hasattr(m, "export") and isinstance(m.export, Callable) and hasattr(m, "_export") and not m._export:
                m.export()

    def _prepare_temporal_samples(self, samples: NestedTensor) -> tuple[NestedTensor, int, int]:
        """Flatten a packed video clip into an image batch for the RF-DETR body.

        The expected clip layout is ``[B, 3*(T+1), H, W]``:
        frame 0 is the key frame with annotations, frames 1..T are reference
        frames.  RF-DETR's existing backbone and transformer then process all
        frames as normal images in one larger batch.
        """

        tensors = samples.tensors  # здесь лежит либо обычное rgb, либо packed clip
        if not self.temporal_enabled:
            return samples, tensors.shape[0], 1  # baseline: один кадр на объект batch

        expected_channels = 3 * (self.temporal_num_ref_frames + 1)  # key frame + reference frames, каждый по 3 канала
        if tensors.shape[1] == 3:
            # Allows normal image inference/eval with a temporal-capable model.
            return samples, tensors.shape[0], 1  # можно прогнать temporal checkpoint на одиночной картинке
        if tensors.shape[1] != expected_channels:
            raise ValueError(
                "Temporal RF-DETR expects packed RGB clips with "
                f"{expected_channels} channels for {self.temporal_num_ref_frames} "
                f"reference frames, got {tensors.shape[1]}."
            )

        batch_size, _, height, width = tensors.shape  # исходный batch клипов
        num_frames = self.temporal_num_ref_frames + 1  # общее число кадров в клипе
        frame_tensors = tensors.view(batch_size, num_frames, 3, height, width).flatten(0, 1)  # превращаем клип в batch кадров

        if samples.mask is None:
            frame_mask = None  # если padding mask не было, backbone ее не получит
        else:
            frame_mask = (
                samples.mask[:, None]  # добавляем ось кадров
                .expand(batch_size, num_frames, height, width)  # повторяем mask для каждого кадра клипа
                .reshape(batch_size * num_frames, height, width)  # flatten так же, как frame_tensors
            )

        return NestedTensor(frame_tensors, frame_mask), batch_size, num_frames  # возвращаем обычный batch кадров и размеры клипа

    @staticmethod
    def _select_nested_frames(samples: NestedTensor, frame_indices: torch.Tensor) -> NestedTensor:
        mask = samples.mask[frame_indices] if samples.mask is not None else None  # выбираем mask только нужных кадров
        return NestedTensor(samples.tensors[frame_indices], mask)  # сохраняем формат NestedTensor для backbone

    @staticmethod
    def _merge_temporal_tensor(key_tensor: torch.Tensor, ref_tensor: torch.Tensor, num_frames: int) -> torch.Tensor:
        if num_frames < 2:
            return key_tensor  # если reference кадров нет, возвращаем обычный tensor
        batch_size = key_tensor.shape[1]  # batch key кадров
        ref_tensor = ref_tensor.view(
            ref_tensor.shape[0],
            batch_size,
            num_frames - 1,  # отделяем reference кадры от batch
            *ref_tensor.shape[2:],
        )
        return torch.cat([key_tensor[:, :, None], ref_tensor], dim=2).flatten(1, 2)  # собираем порядок key, refs обратно

    @staticmethod
    def _merge_temporal_tensor_from_ref_list(
        key_tensor: torch.Tensor,
        ref_tensors: list[torch.Tensor],
    ) -> torch.Tensor:
        if not ref_tensors:
            return key_tensor  # нечего склеивать, если reference список пуст
        refs = torch.stack(ref_tensors, dim=2)  # превращаем список ref tensors в ось кадров
        return torch.cat([key_tensor[:, :, None], refs], dim=2).flatten(1, 2)  # сохраняем тот же порядок кадров

    def _run_detector_body(
        self,
        samples: NestedTensor,
        refpoint_embed_weight: torch.Tensor,
        query_feat_weight: torch.Tensor,
    ):
        features, poss = self.backbone(samples)  # общая часть rf-detr: backbone строит признаки и positional encodings

        srcs = []  # признаки разных уровней для transformer
        masks = []  # padding masks для этих уровней
        for feat in features:
            src, mask = feat.decompose()
            srcs.append(src)  # добавляем feature map
            masks.append(mask)  # добавляем соответствующую mask
            assert mask is not None

        hs, ref_unsigmoid, hs_enc, ref_enc = self.transformer(
            srcs, masks, poss, refpoint_embed_weight, query_feat_weight
        )  # transformer возвращает decoder query и reference boxes
        return features, hs, ref_unsigmoid, hs_enc, ref_enc

    def forward(self, samples: NestedTensor, targets=None):
        """The forward expects a NestedTensor, which consists of:
           - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
           - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

        It returns a dict with the following elements:
           - "pred_logits": the classification logits (including no-object) for all queries.
                            Shape= [batch_size x num_queries x num_classes]
           - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                           (center_x, center_y, width, height). These values are normalized in [0, 1],
                           relative to the size of each individual image (disregarding possible padding).
                           See PostProcess for information on how to retrieve the unnormalized bounding box.
           - "aux_outputs": Optional, only returned when auxiliary losses are activated. It is a list of
                            dictionaries containing the two above keys for each decoder layer.
        """
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        samples, temporal_batch_size, temporal_num_frames = self._prepare_temporal_samples(samples)  # разворачиваем clip в batch кадров

        if self.training:
            refpoint_embed_weight = self.refpoint_embed.weight
            query_feat_weight = self.query_feat.weight
        else:
            # only use one group in inference
            refpoint_embed_weight = self.refpoint_embed.weight[: self.num_queries]
            query_feat_weight = self.query_feat.weight[: self.num_queries]

        if self.segmentation_head is not None:
            seg_head_fwd = self.segmentation_head.sparse_forward if self.training else self.segmentation_head.forward

        features_are_key_only = False  # нужно понять, содержат ли features только key кадр или все кадры
        if self.training and temporal_num_frames > 1:
            frame_indices = torch.arange(  # индексы кадров после flatten: [b*(t+1)]
                samples.tensors.shape[0],
                device=samples.tensors.device,
            ).view(temporal_batch_size, temporal_num_frames)
            key_indices = frame_indices[:, 0]  # первый кадр каждого клипа - обучаемый key frame

            key_samples = self._select_nested_frames(samples, key_indices)  # key кадры идут с градиентами
            features, hs_key, ref_unsigmoid_key, hs_enc_key, ref_enc_key = self._run_detector_body(
                key_samples,
                refpoint_embed_weight,
                query_feat_weight,
            )
            features_are_key_only = True  # features уже относятся только к key кадру

            # Reference frames provide temporal context only.  Running them
            # without gradients keeps the memory footprint close to image
            # training while still giving TemporalQueryFusion real video cues.
            ref_frame_indices = frame_indices[:, 1:]  # матрица reference индексов по каждому клипу
            hs_ref_list = []  # decoder query соседних кадров
            ref_unsigmoid_ref_list = []  # reference boxes соседних кадров
            hs_enc_ref_list = []  # encoder outputs соседних кадров
            ref_enc_ref_list = []  # encoder boxes соседних кадров
            with torch.no_grad():
                for ref_offset in range(temporal_num_frames - 1):
                    ref_samples = self._select_nested_frames(samples, ref_frame_indices[:, ref_offset])  # берем один reference кадр на клип
                    _, hs_ref, ref_unsigmoid_ref, hs_enc_ref, ref_enc_ref = self._run_detector_body(
                        ref_samples,
                        refpoint_embed_weight,
                        query_feat_weight,
                    )  # получаем признаки reference кадра без накопления градиентов
                    hs_ref_list.append(hs_ref)  # сохраняем query для temporal fusion
                    ref_unsigmoid_ref_list.append(ref_unsigmoid_ref)  # сохраняем reference boxes
                    hs_enc_ref_list.append(hs_enc_ref)  # сохраняем encoder output
                    ref_enc_ref_list.append(ref_enc_ref)  # сохраняем encoder boxes

            hs = self._merge_temporal_tensor_from_ref_list(hs_key, hs_ref_list)  # собираем key/ref query в формат [l, b*t, n, c]
            ref_unsigmoid = self._merge_temporal_tensor_from_ref_list(
                ref_unsigmoid_key,
                ref_unsigmoid_ref_list,
            )  # собираем reference boxes decoder
            hs_enc = self._merge_temporal_tensor_from_ref_list(
                hs_enc_key[None],
                [item[None] for item in hs_enc_ref_list],
            )[0]  # собираем encoder outputs и возвращаем исходную размерность
            ref_enc = self._merge_temporal_tensor_from_ref_list(
                ref_enc_key[None],
                [item[None] for item in ref_enc_ref_list],
            )[0]  # собираем encoder reference boxes
        else:
            features, hs, ref_unsigmoid, hs_enc, ref_enc = self._run_detector_body(
                samples,
                refpoint_embed_weight,
                query_feat_weight,
            )  # обычный путь inference/baseline или одиночный кадр

        if hs is not None:
            if self.temporal_query_fusion is not None and temporal_num_frames > 1:
                hs = self.temporal_query_fusion(hs, temporal_batch_size, temporal_num_frames)  # уточняем key query по reference query
                ref_unsigmoid = ref_unsigmoid.view(
                    ref_unsigmoid.shape[0],
                    temporal_batch_size,
                    temporal_num_frames,
                    ref_unsigmoid.shape[-2],
                    ref_unsigmoid.shape[-1],
                )[:, :, 0]  # после fusion оставляем bbox anchors только key кадра
            segmentation_features = features[0].tensors  # segmentation head должен работать с key feature map
            if temporal_num_frames > 1 and not features_are_key_only:
                segmentation_features = segmentation_features.view(
                    temporal_batch_size,
                    temporal_num_frames,
                    segmentation_features.shape[1],
                    segmentation_features.shape[2],
                    segmentation_features.shape[3],
                )[:, 0]  # если features содержали все кадры, берем только key кадр
            if self.bbox_reparam:
                outputs_coord_delta = self.bbox_embed(hs)
                outputs_coord_cxcy = outputs_coord_delta[..., :2] * ref_unsigmoid[..., 2:] + ref_unsigmoid[..., :2]
                outputs_coord_wh = outputs_coord_delta[..., 2:].exp() * ref_unsigmoid[..., 2:]
                outputs_coord = torch.concat([outputs_coord_cxcy, outputs_coord_wh], dim=-1)
            else:
                outputs_coord = (self.bbox_embed(hs) + ref_unsigmoid).sigmoid()

            outputs_class = self.class_embed(hs)

            if self.segmentation_head is not None:
                outputs_masks = seg_head_fwd(segmentation_features, hs, samples.tensors.shape[-2:])

            out = {"pred_logits": outputs_class[-1], "pred_boxes": outputs_coord[-1]}
            if self.segmentation_head is not None:
                out["pred_masks"] = outputs_masks[-1]
            if self.aux_loss:
                out["aux_outputs"] = self._set_aux_loss(
                    outputs_class,
                    outputs_coord,
                    outputs_masks if self.segmentation_head is not None else None,
                )

        if self.two_stage:
            group_detr = self.group_detr if self.training else 1
            if temporal_num_frames > 1:
                hs_enc = hs_enc.view(temporal_batch_size, temporal_num_frames, hs_enc.shape[-2], hs_enc.shape[-1])[:, 0]  # two-stage encoder output нужен только для key кадра
                ref_enc = ref_enc.view(temporal_batch_size, temporal_num_frames, ref_enc.shape[-2], ref_enc.shape[-1])[
                    :, 0
                ]  # encoder boxes тоже берем только для key кадра
            hs_enc_list = hs_enc.chunk(group_detr, dim=1)
            cls_enc = []
            for g_idx in range(group_detr):
                cls_enc_gidx = self.transformer.enc_out_class_embed[g_idx](hs_enc_list[g_idx])
                cls_enc.append(cls_enc_gidx)

            cls_enc = torch.cat(cls_enc, dim=1)

            if self.segmentation_head is not None:
                segmentation_features = features[0].tensors  # маски считаются по key features
                if temporal_num_frames > 1 and not features_are_key_only:
                    segmentation_features = segmentation_features.view(
                        temporal_batch_size,
                        temporal_num_frames,
                        segmentation_features.shape[1],
                        segmentation_features.shape[2],
                        segmentation_features.shape[3],
                    )[:, 0]  # выкидываем reference features перед segmentation head
                masks_enc = seg_head_fwd(
                    segmentation_features,  # key-frame признаки для encoder masks
                    [
                        hs_enc,
                    ],
                    samples.tensors.shape[-2:],
                    skip_blocks=True,
                )[0]

            if hs is not None:
                out["enc_outputs"] = {"pred_logits": cls_enc, "pred_boxes": ref_enc}
                if self.segmentation_head is not None:
                    out["enc_outputs"]["pred_masks"] = masks_enc
            else:
                out = {"pred_logits": cls_enc, "pred_boxes": ref_enc}
                if self.segmentation_head is not None:
                    out["pred_masks"] = masks_enc

        return out

    def forward_export(self, tensors):
        srcs, _, poss = self.backbone(tensors)
        # only use one group in inference
        refpoint_embed_weight = self.refpoint_embed.weight[: self.num_queries]
        query_feat_weight = self.query_feat.weight[: self.num_queries]

        hs, ref_unsigmoid, hs_enc, ref_enc = self.transformer(
            srcs, None, poss, refpoint_embed_weight, query_feat_weight
        )

        outputs_masks = None

        if hs is not None:
            if self.bbox_reparam:
                outputs_coord_delta = self.bbox_embed(hs)
                outputs_coord_cxcy = outputs_coord_delta[..., :2] * ref_unsigmoid[..., 2:] + ref_unsigmoid[..., :2]
                outputs_coord_wh = outputs_coord_delta[..., 2:].exp() * ref_unsigmoid[..., 2:]
                outputs_coord = torch.concat([outputs_coord_cxcy, outputs_coord_wh], dim=-1)
            else:
                outputs_coord = (self.bbox_embed(hs) + ref_unsigmoid).sigmoid()
            outputs_class = self.class_embed(hs)
            if self.segmentation_head is not None:
                outputs_masks = self.segmentation_head(
                    srcs[0],
                    [
                        hs,
                    ],
                    tensors.shape[-2:],
                )[0]
        else:
            assert self.two_stage, "if not using decoder, two_stage must be True"
            outputs_class = self.transformer.enc_out_class_embed[0](hs_enc)
            outputs_coord = ref_enc
            if self.segmentation_head is not None:
                outputs_masks = self.segmentation_head(
                    srcs[0],
                    [
                        hs_enc,
                    ],
                    tensors.shape[-2:],
                    skip_blocks=True,
                )[0]

        if outputs_masks is not None:
            return outputs_coord, outputs_class, outputs_masks
        else:
            return outputs_coord, outputs_class

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord, outputs_masks):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        if outputs_masks is not None:
            return [
                {"pred_logits": a, "pred_boxes": b, "pred_masks": c}
                for a, b, c in zip(outputs_class[:-1], outputs_coord[:-1], outputs_masks[:-1])
            ]
        else:
            return [{"pred_logits": a, "pred_boxes": b} for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]

    def _get_backbone_encoder_layers(self) -> Optional[nn.ModuleList]:
        """Resolve the list of transformer blocks/layers from backbone[0].encoder.

        Supports multiple backbone architectures:
        - encoder.blocks (standard ViT)
        - encoder.trunk.blocks (aimv2)
        - encoder.encoder.encoder.layer (HuggingFace DinoV2)

        Returns:
            List of transformer layers, or None if not found.
        """
        enc = self.backbone[0].encoder
        if hasattr(enc, "blocks"):
            return enc.blocks
        if hasattr(enc, "trunk") and hasattr(enc.trunk, "blocks"):
            return enc.trunk.blocks
        if hasattr(enc, "encoder") and hasattr(enc.encoder, "encoder") and hasattr(enc.encoder.encoder, "layer"):
            return enc.encoder.encoder.layer
        return None

    def update_drop_path(self, drop_path_rate: float, vit_encoder_num_layers: int) -> None:
        """Update drop_path rates for backbone encoder layers with linear schedule.

        Applies a linear schedule where the first layer has drop_path_rate=0 and the last
        layer has drop_path_rate=drop_path_rate. Intermediate layers are interpolated linearly.

        Args:
            drop_path_rate: Maximum drop path rate (applied to last layer).
            vit_encoder_num_layers: Number of encoder layers to update.
        """
        layers = self._get_backbone_encoder_layers()
        if layers is None:
            return
        n = min(vit_encoder_num_layers, len(layers))
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, n)]
        for i in range(n):
            if hasattr(layers[i], "drop_path") and hasattr(layers[i].drop_path, "drop_prob"):
                layers[i].drop_path.drop_prob = dp_rates[i]

    def update_dropout(self, drop_rate):
        for module in self.transformer.modules():
            if isinstance(module, nn.Dropout):
                module.p = drop_rate


def build_model(args: "BuilderArgs"):
    # the `num_classes` naming here is somewhat misleading.
    # it indeed corresponds to `max_obj_id + 1`, where max_obj_id
    # is the maximum id for a class in your dataset. For example,
    # COCO has a max_obj_id of 90, so we pass `num_classes` to be 91.
    # As another example, for a dataset that has a single class with id 1,
    # you should pass `num_classes` to be 2 (max_obj_id + 1).
    # For more details on this, check the following discussion
    # https://github.com/facebookresearch/detr/issues/108#issuecomment-650269223
    num_classes = args.num_classes + 1
    torch.device(args.device)

    backbone = build_backbone(
        encoder=args.encoder,
        vit_encoder_num_layers=args.vit_encoder_num_layers,
        pretrained_encoder=args.pretrained_encoder,
        window_block_indexes=args.window_block_indexes,
        drop_path=args.drop_path,
        out_channels=args.hidden_dim,
        out_feature_indexes=args.out_feature_indexes,
        projector_scale=args.projector_scale,
        use_cls_token=args.use_cls_token,
        hidden_dim=args.hidden_dim,
        position_embedding=args.position_embedding,
        freeze_encoder=args.freeze_encoder,
        layer_norm=args.layer_norm,
        target_shape=(
            args.shape
            if hasattr(args, "shape")
            else ((args.resolution, args.resolution) if hasattr(args, "resolution") else (640, 640))
        ),
        rms_norm=args.rms_norm,
        backbone_lora=args.backbone_lora,
        force_no_pretrain=args.force_no_pretrain,
        gradient_checkpointing=args.gradient_checkpointing,
        load_dinov2_weights=args.pretrain_weights is None,
        patch_size=args.patch_size,
        num_windows=args.num_windows,
        positional_encoding_size=args.positional_encoding_size,
    )
    if args.encoder_only:
        return backbone[0].encoder, None, None
    if args.backbone_only:
        return backbone, None, None

    args.num_feature_levels = len(args.projector_scale)
    transformer = build_transformer(args)

    segmentation_head = (
        SegmentationHead(
            args.hidden_dim,
            args.dec_layers,
            downsample_ratio=args.mask_downsample_ratio,
        )
        if args.segmentation_head
        else None
    )

    model = LWDETR(
        backbone,
        transformer,
        segmentation_head,
        num_classes=num_classes,
        num_queries=args.num_queries,
        aux_loss=args.aux_loss,
        group_detr=args.group_detr,
        two_stage=args.two_stage,
        lite_refpoint_refine=args.lite_refpoint_refine,
        bbox_reparam=args.bbox_reparam,
        temporal_num_ref_frames=getattr(args, "temporal_num_ref_frames", 0),  # прокидываем число reference кадров в модель
        temporal_fusion_layers=getattr(args, "temporal_fusion_layers", 1),  # прокидываем число слоев fusion
        temporal_dropout=getattr(args, "temporal_dropout", 0.0),  # прокидываем dropout temporal блока
    )
    return model


def build_criterion_and_postprocessors(args: "BuilderArgs"):
    device = torch.device(args.device)
    matcher = build_matcher(args)
    weight_dict = {"loss_ce": args.cls_loss_coef, "loss_bbox": args.bbox_loss_coef}
    weight_dict["loss_giou"] = args.giou_loss_coef
    if args.segmentation_head:
        weight_dict["loss_mask_ce"] = args.mask_ce_loss_coef
        weight_dict["loss_mask_dice"] = args.mask_dice_loss_coef
    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
        if args.two_stage:
            aux_weight_dict.update({k + "_enc": v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ["labels", "boxes", "cardinality"]
    if args.segmentation_head:
        losses.append("masks")

    sum_group_losses = getattr(args, "sum_group_losses", False)
    if args.segmentation_head:
        criterion = SetCriterion(
            args.num_classes + 1,
            matcher=matcher,
            weight_dict=weight_dict,
            focal_alpha=args.focal_alpha,
            losses=losses,
            group_detr=args.group_detr,
            sum_group_losses=sum_group_losses,
            use_varifocal_loss=args.use_varifocal_loss,
            use_position_supervised_loss=args.use_position_supervised_loss,
            ia_bce_loss=args.ia_bce_loss,
            mask_point_sample_ratio=args.mask_point_sample_ratio,
        )
    else:
        criterion = SetCriterion(
            args.num_classes + 1,
            matcher=matcher,
            weight_dict=weight_dict,
            focal_alpha=args.focal_alpha,
            losses=losses,
            group_detr=args.group_detr,
            sum_group_losses=sum_group_losses,
            use_varifocal_loss=args.use_varifocal_loss,
            use_position_supervised_loss=args.use_position_supervised_loss,
            ia_bce_loss=args.ia_bce_loss,
        )
    criterion.to(device)
    postprocess = PostProcess(num_select=args.num_select)

    return criterion, postprocess


def build_model_from_config(
    model_config: "ModelConfig",
    train_config: Optional["TrainConfig"] = None,
    defaults: ModelDefaults = MODEL_DEFAULTS,
) -> LWDETR:
    """Build an LWDETR model directly from a ModelConfig.

    A config-native alternative to ``build_model(build_namespace(mc, tc))``.
    Constructs the namespace internally from ``model_config``, an optional
    ``train_config``, and ``defaults``, then delegates to :func:`build_model`.

    Note:
        The internal ``SimpleNamespace`` bridge is transitional — it will be
        eliminated once all builder functions accept config objects directly.
        Callers should not rely on the namespace shape or pass it externally.

    Args:
        model_config: Architecture configuration.
        train_config: Training hyperparameter configuration. If ``None``,
            a minimal dummy ``TrainConfig(dataset_dir=".", output_dir=".")`` is
            constructed, matching the previous default behavior.
        defaults: Hardcoded architectural constants. Defaults to ``MODEL_DEFAULTS``.

    Returns:
        Fully initialised LWDETR model.

    Raises:
        ValueError: If ``defaults`` request ``encoder_only`` or ``backbone_only``,
            which would make the return type differ from ``LWDETR``.
    """
    from rfdetr._namespace import _namespace_from_configs

    if defaults.encoder_only or defaults.backbone_only:
        raise ValueError(
            "build_model_from_config() requires defaults.encoder_only=False and defaults.backbone_only=False."
        )

    if train_config is None:
        from rfdetr.config import TrainConfig

        train_config = TrainConfig(dataset_dir=".", output_dir=".")

    ns = _namespace_from_configs(model_config, train_config, defaults)
    return build_model(ns)


def build_criterion_from_config(
    model_config: "ModelConfig",
    train_config: "TrainConfig",
    defaults: ModelDefaults = MODEL_DEFAULTS,
) -> tuple[SetCriterion, PostProcess]:
    """Build criterion and postprocessor directly from config objects.

    A config-native alternative to
    ``build_criterion_and_postprocessors(build_namespace(mc, tc))``.

    Args:
        model_config: Architecture configuration.
        train_config: Training hyperparameter configuration.
        defaults: Hardcoded architectural constants. Defaults to ``MODEL_DEFAULTS``.

    Returns:
        A 2-tuple of ``(SetCriterion, PostProcess)``.
    """
    from rfdetr._namespace import _namespace_from_configs

    ns = _namespace_from_configs(model_config, train_config, defaults)
    return build_criterion_and_postprocessors(ns)
