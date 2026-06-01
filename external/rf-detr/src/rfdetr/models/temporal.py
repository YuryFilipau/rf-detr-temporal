# ------------------------------------------------------------------------
# RF-DETR temporal extensions for video object detection.
# Copyright (c) 2026.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Temporal query fusion modules inspired by TransVOD++.

RF-DETR is an image detector: every decoder query describes one candidate
object in the current image.  TransVOD++ adds temporal reasoning mostly at the
query level, after a spatial detector has produced query embeddings for the key
frame and neighbouring reference frames.

This file keeps that idea, but avoids copying TransVOD++'s full Sparse-RCNN/QRF
stack.  The module below is deliberately small and compatible with RF-DETR's
existing decoder outputs:

    current frame queries -> MultiHeadAttention -> reference frame queries

The result is still a normal ``hs`` tensor, so RF-DETR's classification head,
box head, criterion, and post-processing remain unchanged.
"""

from __future__ import annotations

'''
этот файл содержит сам temporal fusion module.
он берет query текущего кадра и через attention смотрит на query соседних
кадров, чтобы сохранить полезный контекст: где машина была раньше или рядом
во времени, даже если на текущем кадре она маленькая, перекрыта или смазана.
'''

from typing import Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class TemporalQueryFusionLayer(nn.Module):
    """Fuse current-frame object queries with reference-frame queries.

    Args:
        d_model: Query feature dimension.
        nhead: Number of attention heads.
        dim_feedforward: Hidden size of the feed-forward block.
        dropout: Dropout probability.
        activation: Feed-forward activation name.

    Shape:
        ``query``: ``[B, N, C]``
        ``ref_query``: ``[B, T, N, C]`` or ``[B, T*N, C]``
        output: ``[B, N, C]``
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.0,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)  # current query смотрит на query соседних кадров
        self.dropout1 = nn.Dropout(dropout)  # dropout после attention
        self.norm1 = nn.LayerNorm(d_model)  # стабилизирует residual после attention

        self.linear1 = nn.Linear(d_model, dim_feedforward)  # первая часть ffn, как в transformer слое
        self.dropout = nn.Dropout(dropout)  # dropout внутри ffn
        self.linear2 = nn.Linear(dim_feedforward, d_model)  # возвращаем размерность обратно в d_model
        self.dropout2 = nn.Dropout(dropout)  # dropout перед вторым residual
        self.norm2 = nn.LayerNorm(d_model)  # нормализация после ffn
        self.activation = _get_activation_fn(activation)  # выбираем relu/gelu/glu

    def forward(
        self,
        query: Tensor,
        ref_query: Tensor,
        query_pos: Optional[Tensor] = None,
        ref_query_pos: Optional[Tensor] = None,
    ) -> Tensor:
        if ref_query.dim() == 4:
            batch_size, num_ref_frames, num_queries, channels = ref_query.shape  # ref задан как [b, t, n, c]
            ref_query = ref_query.reshape(batch_size, num_ref_frames * num_queries, channels)  # attention ждет одну ось tokens
            if ref_query_pos is not None:
                ref_query_pos = ref_query_pos.reshape(batch_size, num_ref_frames * num_queries, channels)  # позиционные признаки выравниваем так же

        q = query if query_pos is None else query + query_pos  # q - текущий кадр
        k = ref_query if ref_query_pos is None else ref_query + ref_query_pos  # k - соседние кадры
        v = ref_query  # v несет сами признаки соседних кадров

        attended = self.cross_attn(q, k, v, need_weights=False)[0]  # текущие query получают информацию из reference query
        query = self.norm1(query + self.dropout1(attended))  # residual сохраняет исходный текущий кадр

        ffn = self.linear2(self.dropout(self.activation(self.linear1(query))))  # ffn дообрабатывает смешанные признаки
        query = self.norm2(query + self.dropout2(ffn))  # второй residual как в transformer блоке
        return query


class TemporalQueryFusion(nn.Module):
    """Apply temporal query fusion to RF-DETR decoder hidden states.

    RF-DETR returns hidden states as ``[L, B*(T+1), N, C]`` when a clip is
    flattened into the batch dimension.  The first frame in every clip is the
    key/current frame, matching TransVOD++ datasets where the target annotation
    belongs to the first frame.  Reference frames provide context only.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
        num_layers: int,
        queries_per_group: Optional[int] = None,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("TemporalQueryFusion requires num_layers >= 1.")
        self.queries_per_group = queries_per_group  # нужно для group-detr: каждая группа query обрабатывается отдельно
        self.layers = nn.ModuleList(
            [
                TemporalQueryFusionLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, hs: Tensor, batch_size: int, num_frames: int) -> Tensor:
        if hs.dim() != 4:
            raise ValueError(f"Expected hs [L, B*(T+1), N, C], got {tuple(hs.shape)}.")
        if num_frames < 2:
            return hs  # нет соседних кадров, значит нечего fusion-ить
        if hs.shape[1] != batch_size * num_frames:
            raise ValueError(
                "TemporalQueryFusion got inconsistent batch/frame dimensions: "
                f"hs batch={hs.shape[1]}, batch_size={batch_size}, num_frames={num_frames}."
            )

        num_decoder_layers, _, num_queries, channels = hs.shape  # l - decoder layers, n - queries
        hs_by_frame = hs.view(num_decoder_layers, batch_size, num_frames, num_queries, channels)  # восстанавливаем ось кадров

        # RF-DETR trains with Group DETR: the decoder query dimension is
        # num_queries * group_detr (for example 300 * 13).  Attending all groups
        # to all reference groups would create a huge dense matrix and does not
        # match the independent-group training design.  Fuse each query group
        # independently, then restore the original query layout for the heads.
        queries_per_group = self.queries_per_group or num_queries  # если групп нет, вся ось query одна группа
        if num_queries % queries_per_group != 0:
            queries_per_group = num_queries  # fallback, если размерность не делится на группы
        num_query_groups = num_queries // queries_per_group  # сколько group-detr групп получилось

        fused_layers = []  # здесь будут query key кадра после fusion для каждого decoder layer
        for layer_idx in range(num_decoder_layers):
            current = hs_by_frame[layer_idx, :, 0]  # query текущего кадра
            refs = hs_by_frame[layer_idx, :, 1:]  # query reference кадров

            if num_query_groups > 1:
                current = current.view(
                    batch_size,
                    num_query_groups,
                    queries_per_group,
                    channels,
                ).flatten(0, 1)  # каждая group-detr группа становится отдельным batch элементом
                refs = (
                    refs.view(
                        batch_size,
                        num_frames - 1,
                        num_query_groups,
                        queries_per_group,
                        channels,
                    )
                    .permute(0, 2, 1, 3, 4)  # ставим group рядом с batch
                    .reshape(
                        batch_size * num_query_groups,
                        num_frames - 1,
                        queries_per_group,
                        channels,
                    )  # reference query той же группы попадут в attention вместе
                )

            for fusion_layer in self.layers:
                current = fusion_layer(current, refs)  # применяем один или несколько temporal attention слоев

            if num_query_groups > 1:
                current = current.view(
                    batch_size,
                    num_query_groups,
                    queries_per_group,
                    channels,
                ).reshape(batch_size, num_queries, channels)  # возвращаем исходную ось query
            fused_layers.append(current)  # сохраняем key-frame query этого decoder layer

        return torch.stack(fused_layers, dim=0)  # итог снова [l, b, n, c], как ожидают головы rf-detr


def _get_activation_fn(activation: str):
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu/glu, not {activation}.")
