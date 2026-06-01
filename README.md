# RF-DETR Temporal

Модификация RF-DETR для работы с последовательностями кадров. Базовый RF-DETR обучается на отдельных изображениях. В этой версии к текущему кадру можно добавить соседние кадры из той же видеопоследовательности и использовать их в decoder query fusion.

## Основная идея

Датасет хранится в COCO-формате, но для каждого изображения дополнительно используются поля:

```text
video_id
frame_id
```

По ним датасет находит соседние кадры и собирает клип:

```text
key frame + reference frames
```

Аннотации относятся только к key frame. Reference frames используются только как контекст.

Внутри модели RF-DETR сначала получает query-представления для key frame и reference frames. Затем `TemporalQueryFusion` уточняет query key frame через attention к query соседних кадров. После этого используются обычные RF-DETR class/bbox heads.

## Измененные файлы

```text
external/rf-detr/src/rfdetr/models/temporal.py
```

Новый модуль temporal fusion. Реализует attention между query текущего кадра и query соседних кадров.

```text
external/rf-detr/src/rfdetr/models/lwdetr.py
```

Интеграция temporal режима в RF-DETR:

- разбор packed clip;
- отделение key frame от reference frames;
- прогон reference frames без градиентов при обучении;
- применение temporal fusion перед bbox/class heads;
- сохранение совместимости с обычным RF-DETR.

```text
external/rf-detr/src/rfdetr/datasets/coco.py
```

Добавлен `TemporalCocoDetection`. Он собирает соседние кадры по `video_id/frame_id` или по имени файла, если `video_id` отсутствует.

```text
external/rf-detr/src/rfdetr/config.py
external/rf-detr/src/rfdetr/_namespace.py
external/rf-detr/src/rfdetr/models/_types.py
```

Добавлены параметры temporal режима и их передача в модель.

## Структура

```text
rf_detr_temporal/
├── commands/
│   ├── train_baseline.sh
│   ├── train_temporal.sh
│   └── train_baseline_then_temporal.sh
├── dataset/
│   ├── annotations/
│   │   ├── instances_train2017.json
│   │   └── instances_val2017.json
│   └── build_manifest.json
├── external/rf-detr/
├── reports/rfdetr_full_10ep_comparison/
├── results_10ep/
│   ├── baseline/
│   ├── temporal/
│   └── logs/
└── tools/
```

## Датасет

Основной датасет:

```text
data/car_dataset_temporal/
├── train2017/
├── val2017/
└── annotations/
    ├── instances_train2017.json
    └── instances_val2017.json
```

В эту папку export скопированы только аннотации и `build_manifest.json`. Изображения и видео не включены из-за размера.

Состав:

| split | images | annotations | empty frames |
|---|---:|---:|---:|
| train | 8177 | 20869 | 209 |
| val | 2143 | 3664 | 0 |

Разделение train/val сделано по целым видео.

## Команды

Запуск baseline:

```bash
bash commands/train_baseline.sh
```

Запуск temporal:

```bash
bash commands/train_temporal.sh
```

Запуск двух экспериментов подряд:

```bash
bash commands/train_baseline_then_temporal.sh
```

## Baseline

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=0 /home/yury/anaconda3/envs/TransVOD++/bin/python tools/train_rfdetr_baseline.py \
  --dataset-root data/car_dataset_temporal \
  --output-dir exps/rfdetr_full_baseline_10ep \
  --variant small \
  --num-classes 1 \
  --resolution 640 \
  --epochs 10 \
  --batch-size 1 \
  --grad-accum-steps 4 \
  --lr 1e-4 \
  --lr-encoder 1e-5 \
  --lr-drop 8 \
  --num-workers 2
```

## Temporal

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=0 /home/yury/anaconda3/envs/TransVOD++/bin/python tools/train_rfdetr_temporal.py \
  --dataset-root data/car_dataset_temporal \
  --output-dir exps/rfdetr_full_temporal_surrounding3_step2_10ep \
  --variant small \
  --num-classes 1 \
  --resolution 640 \
  --temporal-num-ref-frames 3 \
  --temporal-fusion-layers 1 \
  --temporal-ref-frame-mode surrounding \
  --temporal-ref-frame-step 2 \
  --epochs 10 \
  --batch-size 1 \
  --grad-accum-steps 4 \
  --lr 1e-4 \
  --lr-encoder 1e-5 \
  --lr-drop 8 \
  --num-workers 2
```

Параметры temporal:

- `--temporal-num-ref-frames 3` - количество соседних кадров.
- `--temporal-fusion-layers 1` - число слоев temporal attention.
- `--temporal-ref-frame-mode surrounding` - брать кадры вокруг текущего.
- `--temporal-ref-frame-step 2` - шаг между кадрами.

## Результаты 10 эпох

| model | mAP 50:95 | EMA mAP 50:95 | AP50 | AP75 | mAR | val loss |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 0.7523 | 0.7540 | 0.9363 | 0.8290 | 0.8052 | 2.9813 |
| temporal | 0.7612 | 0.7621 | 0.9430 | 0.8407 | 0.8025 | 2.8761 |

Файлы:

```text
results_10ep/
reports/rfdetr_full_10ep_comparison/
```

Temporal версия получила немного выше mAP, AP50, AP75 и ниже validation loss. Baseline получил немного выше mAR. Для окончательного вывода нужно смотреть не только COCO-метрики, но и поведение на видео: пропуски, скачки bbox и стабильность score между соседними кадрами.

## Датасет

Датасет, использованный для обучения моделей: https://drive.google.com/drive/folders/1O9LU4XEXSAF2sSBWG3jrWRjRuYtToKOA?usp=drive_link
