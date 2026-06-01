#!/usr/bin/env python
"""Smoke checks for the TransVOD-style temporal RF-DETR extension.

This script is intentionally lightweight: it does not download DINOv2/RF-DETR
weights and does not require a GPU.  It verifies the temporal query-fusion
contract and the packed-clip path through ``LWDETR`` using tiny dummy backbone
and transformer modules.

Run from the RF-DETR repository root after installing dependencies:

    PYTHONPATH=src python tools/smoke_temporal_rfdetr.py
"""

from __future__ import annotations

'''
этот файл быстро проверяет, что temporal часть rf-detr вообще собирается.
он не обучает модель и не качает веса, а прогоняет маленькие фиктивные данные,
чтобы поймать ошибки формы тензоров до долгого запуска обучения.
'''

import torch
from torch import nn

from rfdetr.models.lwdetr import LWDETR
from rfdetr.models.temporal import TemporalQueryFusion
from rfdetr.utilities.tensors import NestedTensor


class _DummyBackbone(nn.Module):
    def forward(self, samples: NestedTensor):
        tensors = samples.tensors
        batch_size = tensors.shape[0]
        device = tensors.device
        feat = torch.randn(batch_size, 32, 4, 4, device=device)
        mask = torch.zeros(batch_size, 4, 4, dtype=torch.bool, device=device)
        pos = torch.zeros(batch_size, 32, 4, 4, device=device)
        return [NestedTensor(feat, mask)], [pos]


class _DummyDecoder(nn.Module):
    def __init__(self, hidden_dim: int, nheads: int) -> None:
        super().__init__()
        layer = nn.Module()
        layer.self_attn = nn.MultiheadAttention(hidden_dim, nheads, batch_first=True)
        layer.linear1 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.layers = nn.ModuleList([layer])
        self.bbox_embed = None


class _DummyTransformer(nn.Module):
    def __init__(self, hidden_dim: int = 32, num_queries: int = 5, num_layers: int = 2) -> None:
        super().__init__()
        self.d_model = hidden_dim
        self.num_queries = num_queries
        self.num_layers = num_layers
        self.decoder = _DummyDecoder(hidden_dim, nheads=4)

    def forward(self, srcs, masks, poss, refpoint_embed_weight, query_feat_weight):
        batch_size = srcs[0].shape[0]
        hidden_dim = self.d_model
        hs = torch.randn(self.num_layers, batch_size, self.num_queries, hidden_dim, device=srcs[0].device)
        ref = torch.zeros(self.num_layers, batch_size, self.num_queries, 4, device=srcs[0].device)
        return hs, ref, None, None


def main() -> None:
    torch.manual_seed(0)

    fusion = TemporalQueryFusion(d_model=32, nhead=4, dim_feedforward=64, dropout=0.0, num_layers=1)
    hs = torch.randn(2, 6, 5, 32)
    fused = fusion(hs, batch_size=2, num_frames=3)
    assert fused.shape == (2, 2, 5, 32), fused.shape
    print("TemporalQueryFusion:", tuple(fused.shape))

    model = LWDETR(
        backbone=_DummyBackbone(),
        transformer=_DummyTransformer(),
        segmentation_head=None,
        num_classes=2,
        num_queries=5,
        aux_loss=True,
        group_detr=1,
        two_stage=False,
        temporal_num_ref_frames=2,
        temporal_fusion_layers=1,
    )
    samples = NestedTensor(
        torch.randn(2, 9, 32, 32),
        torch.zeros(2, 32, 32, dtype=torch.bool),
    )
    out = model(samples)
    assert out["pred_logits"].shape == (2, 5, 2), out["pred_logits"].shape
    assert out["pred_boxes"].shape == (2, 5, 4), out["pred_boxes"].shape
    print("LWDETR temporal forward:", tuple(out["pred_logits"].shape), tuple(out["pred_boxes"].shape))


if __name__ == "__main__":
    main()
