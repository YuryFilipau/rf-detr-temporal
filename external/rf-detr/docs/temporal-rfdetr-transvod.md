# Temporal RF-DETR: TransVOD++ Query Fusion

This fork adds a conservative temporal path to RF-DETR for video object
detection in autonomous-driving style scenes.

## Architecture

Standard RF-DETR remains unchanged for image inputs:

```text
image -> RF-DETR backbone/projector -> RF-DETR decoder -> class/box heads
```

When `temporal_num_ref_frames > 0`, the model also accepts packed RGB clips:

```text
[key, ref1, ref2, ...] -> B x 3*(T+1) x H x W
```

The key frame is always frame `0`; targets belong only to this frame. Internally:

1. `LWDETR._prepare_temporal_samples` reshapes the clip into `B*(T+1)` normal
   images.
2. The unmodified RF-DETR backbone and decoder process all frames.
3. `TemporalQueryFusion` fuses decoder queries:

```text
current-frame queries attend to reference-frame queries
```

4. The original RF-DETR class and box heads run on the fused current-frame
queries.

This is the most portable TransVOD++ idea: temporal query reasoning. It avoids
copying TransVOD++'s Sparse-RCNN/QRF stack and keeps RF-DETR losses,
post-processing, checkpoint loading, and export boundaries understandable.

## Dataset Path

`TemporalCocoDetection` returns packed clips while keeping the original current
frame target:

```text
image:  [3*(T+1), H, W]
target: boxes/labels for key frame only
```

Reference-frame modes:

- `previous`: causal, recommended for autopilot deployment.
- `surrounding`: can use future frames, useful only for offline experiments.
- `duplicate`: repeats the key frame; use as a no-motion ablation.

For temporal training, RF-DETR uses deterministic resize/normalize transforms
for all frames. This avoids corrupting temporal consistency by applying random
crops/flips independently to key and reference frames.

## Recommended First Experiments

Use `previous` references first:

```python
from rfdetr import RFDETRSmall

model = RFDETRSmall(
    num_classes=1,
    temporal_num_ref_frames=3,
    temporal_fusion_layers=1,
)

model.train(
    dataset_file="coco",
    dataset_dir="/home/yury/PycharmProjects/TransVOD_plusplus/data/dandelion_dataset2",
    output_dir="output/rfdetr_temporal_prev3",
    temporal_ref_frame_mode="previous",
    temporal_ref_frame_step=1,
    batch_size=1,
    grad_accum_steps=4,
    epochs=30,
)
```

Run a control with duplicated references:

```python
model.train(
    ...,
    temporal_ref_frame_mode="duplicate",
)
```

If `previous` beats `duplicate`, the temporal module is learning motion/context
rather than merely acting as extra parameters.

## What To Watch

- `mAP50:95`, `mAP50`, and recall for `small` objects. Driving hazards often
  start as small/far objects.
- False positives on background vehicles, shadows, road texture, and night/fog
  scenes.
- Frame-to-frame stability: predictions should jitter less with temporal fusion.
- Causal vs surrounding gap. If `surrounding` is much better, the model may rely
  on future information and should not be treated as deployable autopilot logic.
- Runtime and VRAM: temporal RF-DETR processes `T+1` frames through the backbone,
  so memory roughly scales with the number of frames before query fusion.

## Smoke Test

After installing RF-DETR dependencies:

```bash
cd external/rf-detr
PYTHONPATH=src python tools/smoke_temporal_rfdetr.py
```

Expected output:

```text
TemporalQueryFusion: (2, 2, 5, 32)
LWDETR temporal forward: (2, 5, 2) (2, 5, 4)
```
