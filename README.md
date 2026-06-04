# RF-DETR Temporal

Модификация RF-DETR для детекции автомобилей на последовательностях кадров. Обычный RF-DETR работает как image detector: один входной кадр, один набор признаков, один набор query, один прогноз bbox/class. В этой работе добавлен temporal-контекст, то есть, вместе с текущим кадром модель может использовать соседние кадры из того же видео.

## Краткое сравнение.

Базовый RF-DETR:

```text
image -> backbone -> transformer decoder queries -> class/bbox heads
```

RF-DETR Temporal:

```text
key frame + reference frames
-> backbone/transformer для каждого кадра
-> query key frame + query reference frames
-> temporal query fusion
-> обычные class/bbox heads RF-DETR
```

Аннотации используются только для key frame. Reference frames не имеют своего loss. Они нужны как дополнительный контекст.

## Архитектура
<img width="1280" height="525" alt="image_2026-06-04_13-16-39" src="https://github.com/user-attachments/assets/0d4fc167-9026-4014-bf98-827ae2489f65" />

### Input Frames

На вход подается набор кадров из одной видеопоследовательности:

```text
Key frame t
Reference frame t-2
Reference frame t+2
Reference frame t+4
```

`Key frame` - основной кадр, для которого есть ground truth и по которому считается loss.

`Reference frames` - соседние кадры. Они проходят через модель, но не имеют собственного target loss. Их задача - дать temporal-контекст для текущего кадра.

### Temporal COCO Loader

`Temporal COCO Loader` выбирает соседние кадры по полям:

```text
video_id
frame_id
```

Это позволяет собрать кадры из одного и того же видео и сохранить правильный временной порядок.

### Packed Clip

После загрузки кадры объединяются в один packed clip:

```text
[B, 3*(T+1), H, W]
```

Где:

```text
B - batch size
T - количество reference frames
3 - RGB каналы одного кадра
H, W - высота и ширина изображения
```

Например, если используется 3 reference frames, то вход имеет 12 каналов:

```text
3 * (3 + 1) = 12
```

### Unpack Frames

RF-DETR backbone ожидает обычные RGB изображения с 3 каналами. Поэтому packed clip разворачивается обратно в batch отдельных кадров:

```text
[B, 3*(T+1), H, W] -> [B*(T+1), 3, H, W]
```

После этого каждый кадр может быть обработан обычным RF-DETR backbone.

### Shared RF-DETR Processing

Каждый кадр проходит через одинаковую RF-DETR часть:

```text
Backbone -> Encoder / Decoder
```

Для всех кадров используются одни и те же веса модели.

В результате формируются два типа query:

```text
Key-frame Queries: [B, N, C]
Reference-frame Queries: [B, T, N, C]
```

Где:

```text
N - количество object queries
C - размерность query-признаков
```

### Temporal Query Fusion

`Temporal Query Fusion` объединяет признаки текущего кадра и соседних кадров.

Key-frame queries выступают как основной объект уточнения, а reference-frame queries используются как источник временного контекста.

### Group-DETR Query Fusion

RF-DETR использует Group-DETR queries. Чтобы снизить расход памяти, temporal fusion выполняется по группам queries, а не для всех queries сразу.

### Detection Heads

После temporal fusion остаются только уточненные query текущего кадра:

```text
Fused Key-frame Queries
```

Они передаются в стандартные RF-DETR heads:

```text
Class / Box Heads
```

Архитектура heads не меняется.

### Output

На выходе модель предсказывает объекты только для key frame, а Loss считается только по аннотациям текущего кадра.

Reference frames не сравниваются с ground truth напрямую. Их задача - улучшить признаки текущего кадра за счет временного контекста.

## Зачем это сделано

Для задачи автопилота объект на одном кадре может быть плохо виден:

- автомобиль маленький;
- автомобиль частично перекрыт;
- bbox меняется из-за размытия;
- на одном кадре объект почти сливается с фоном;
- confidence скачет между соседними кадрами;
- модель то находит объект, то пропускает его.

Если модель видит соседние кадры, она может уточнить признаки текущего кадра. Например, если автомобиль на текущем кадре частично закрыт, но на предыдущем или следующем кадре виден лучше, query текущего кадра может получить дополнительный контекст через attention.

## Данные

Датасет остается COCO-like:

```text
data/car_dataset_temporal/
├── train2017/
├── val2017/
└── annotations/
    ├── instances_train2017.json
    └── instances_val2017.json
```

В COCO `images` для temporal режима важны поля:

```text
video_id
frame_id
file_name
width
height
```

`video_id` нужен, чтобы понять, какие кадры принадлежат одному видео. `frame_id` или порядок имени файла нужен, чтобы выбрать соседние кадры. Если `video_id` отсутствует, код пытается восстановить последовательность по имени файла.

Состав актуального датасета:

| split | images | annotations | empty frames |
|---|---:|---:|---:|
| train | 8177 | 20869 | 209 |
| val | 2143 | 3664 | 0 |

Train/val разделены по целым видео. Это важно: если кадры одного видео попадут одновременно в train и val, оценка будет слишком оптимистичной.

В export скопированы только аннотации и `build_manifest.json`. Полные изображения, видео и веса не включены из-за размера.

## Как собирается temporal clip

Для каждого key frame датасет выбирает reference frames. Вход модели упаковывается по каналам:

```text
[B, 3 * (T + 1), H, W]
```

Где:

- `B` - batch size;
- `T` - число reference frames;
- `3` - RGB каналы одного кадра;
- первый блок из 3 каналов - key frame;
- остальные блоки по 3 канала - reference frames.

Пример при `--temporal-num-ref-frames 3`:

```text
[B, 12, H, W]
```

Порядок:

```text
key_rgb + ref_1_rgb + ref_2_rgb + ref_3_rgb
```

Такой формат выбран потому, что RF-DETR уже умеет принимать тензор изображения. Перед backbone этот packed clip разворачивается обратно в batch отдельных кадров.

## Режимы выбора reference frames

В temporal датасете поддерживаются три режима.

```text
previous
```

Берутся только предыдущие кадры.

```text
surrounding
```

Берутся кадры вокруг текущего: предыдущие и следующие. Так же реализован TransVOD.

```text
duplicate
```

Текущий кадр дублируется как reference. Это контрольный режим: temporal модуль включен, но настоящего движения нет. Его можно использовать, чтобы проверить, дает ли прирост именно temporal-контекст, а не просто дополнительные вычисления.

Параметр:

```text
--temporal-ref-frame-step
```

задает расстояние между кадрами. При `step=2` берутся не соседние кадры подряд, а через один. Это полезно, если соседние кадры слишком похожи(применялось при тестировании).

## Архитектура изменений

Модификация не заменяет весь RF-DETR. Основная структура RF-DETR сохранена:

- backbone RF-DETR остается прежним;
- transformer RF-DETR остается прежним;
- class head остается прежним;
- bbox head остается прежним;
- criterion/loss остаются прежними;
- COCO evaluation остается прежним.

Изменение добавлено между decoder output и heads:

```text
decoder hs -> temporal fusion -> class/bbox heads
```

Это сделано для совместимости: после temporal fusion выход снова имеет формат обычных decoder hidden states RF-DETR.

## Новый файл: `external/rf-detr/src/rfdetr/models/temporal.py`

Файл добавлен полностью. В нем находится temporal fusion module.

### `TemporalQueryFusionLayer`

Один слой temporal fusion. Его задача - взять query текущего кадра и дать им посмотреть на query соседних кадров через `nn.MultiheadAttention`.

Вход:

```text
query:     [B, N, C]
ref_query: [B, T, N, C] или [B, T*N, C]
```

Где:

- `B` - batch size;
- `T` - число reference frames;
- `N` - число object queries;
- `C` - hidden dimension.

Внутри:

```text
q = query
k = ref_query
v = ref_query
```

То есть текущий кадр спрашивает у соседних кадров, какие признаки могут быть полезны для его query.

После attention используется residual connection:

```text
query = norm(query + attended)
```

Затем идет feed-forward block:

```text
linear -> activation -> dropout -> linear -> residual -> norm
```

Это похоже на стандартный transformer layer, но attention направлен не внутрь одного кадра, а между текущим и соседними кадрами.

### `TemporalQueryFusion`

Этот класс применяет один или несколько `TemporalQueryFusionLayer` к выходу decoder.

RF-DETR decoder возвращает:

```text
hs: [L, B*(T+1), N, C]
```

Где:

- `L` - число decoder layers;
- `B*(T+1)` - batch после разворачивания клипа в отдельные кадры;
- `N` - число queries;
- `C` - hidden dimension.

Сначала код восстанавливает структуру:

```text
[L, B, T+1, N, C]
```

Потом отделяет:

```text
current = hs[:, :, 0]
refs = hs[:, :, 1:]
```

После fusion сохраняется только текущий кадр:

```text
output: [L, B, N, C]
```

Это важно: loss и heads должны работать только с key frame.

### Group-DETR обработка

RF-DETR при обучении использует Group-DETR. Например, при `num_queries=300` и `group_detr=13` фактическая ось query может быть:

```text
300 * 13 = 3900 queries
```

Если сделать attention всех query ко всем reference query сразу, память резко вырастет. Поэтому в `TemporalQueryFusion` query делятся на группы:

```text
[B, group_detr, queries_per_group, C]
```

Каждая группа проходитfusion отдельно. Это снижает расход памяти и соответствует логике Group-DETR, где группы обучаются независимо.

## Измененный файл: `external/rf-detr/src/rfdetr/models/lwdetr.py`

Это основной файл модели RF-DETR. В него добавлена интеграция temporal режима.

### Новый импорт

Добавлен:

```python
from rfdetr.models.temporal import TemporalQueryFusion
```

Он подключает новый temporal module.

### Новые параметры `LWDETR.__init__`

Добавлены:

```python
temporal_num_ref_frames=0
temporal_fusion_layers=1
temporal_dropout=0.0
```

Назначение:

- `temporal_num_ref_frames` - сколько соседних кадров ожидает модель;
- `temporal_fusion_layers` - сколько temporal attention слоев создать;
- `temporal_dropout` - dropout внутри temporal fusion.

Если `temporal_num_ref_frames == 0`, temporal отключен, модель работает как обычный RF-DETR.

### Новый флаг `self.temporal_enabled`

```python
self.temporal_enabled = self.temporal_num_ref_frames > 0
```

Флаг нужен, чтобы baseline и temporal могли использовать один и тот же код модели. Baseline не должен заходить в temporal путь.

### Новый модуль `self.temporal_query_fusion`

Создается только если temporal включен:

```python
self.temporal_query_fusion = TemporalQueryFusion(...)
```

Параметры берутся из уже существующего decoder:

- `d_model=hidden_dim`;
- `nhead=transformer.decoder.layers[0].self_attn.num_heads`;
- `dim_feedforward=transformer.decoder.layers[0].linear1.out_features`.

Так temporal module совпадает по размерностям с transformer RF-DETR.

### `_prepare_temporal_samples`

Новый метод. Он разбирает packed clip:

```text
[B, 3*(T+1), H, W]
```

и превращает его в batch кадров:

```text
[B*(T+1), 3, H, W]
```

Причина: backbone и transformer RF-DETR уже умеют работать с batch изображений. Поэтому проще временно представить клип как большой batch кадров.

Метод также обрабатывает mask:

```text
[B, H, W] -> [B*(T+1), H, W]
```

Если модель temporal-capable, но на вход пришло обычное RGB изображение `[B, 3, H, W]`, код разрешает такой режим и работает как обычный RF-DETR. Это нужно для совместимости inference/eval.

### `_select_nested_frames`

Новый helper для выбора key или reference кадров из `NestedTensor`.

Он нужен, потому что RF-DETR работает не с обычным tensor, а с `NestedTensor`, где есть:

- `samples.tensors`;
- `samples.mask`.

Метод выбирает одинаковые индексы и из tensor, и из mask.

### `_merge_temporal_tensor_from_ref_list`

Новый helper для обратной сборки:

```text
key tensor + list(reference tensors) -> [key, refs] в общей оси batch/frame
```

Он нужен после отдельного прогона key frame и reference frames.

### `_run_detector_body`

Новый helper, который содержит общую часть RF-DETR:

```text
backbone -> transformer
```

До модификации эта логика находилась прямо внутри `forward`. После добавления temporal режима ее вынесли в отдельный метод, чтобы использовать одинаково:

- для key frame;
- для reference frames;
- для обычного baseline/inference пути.

### Изменение `forward`

В начале `forward` добавлено:

```python
samples, temporal_batch_size, temporal_num_frames = self._prepare_temporal_samples(samples)
```

Это приводит вход к формату, который понимает RF-DETR body.

Дальше есть два режима.

#### Training + temporal

Если модель обучается и `temporal_num_frames > 1`, код делает:

1. Разделяет индексы кадров:

```text
key_indices = frame_indices[:, 0]
ref_frame_indices = frame_indices[:, 1:]
```

2. Key frames прогоняются с градиентами:

```text
key_samples -> backbone -> transformer
```

3. Reference frames прогоняются без градиентов:

```python
with torch.no_grad():
    ref_samples -> backbone -> transformer
```

Это сделано из-за ограничения видеопамяти. Если считать градиенты для всех reference frames, расход памяти сильно растет. В текущей реализации reference frames дают контекст, но напрямую не обучаются через loss.

4. Query key frame и reference frames собираются обратно:

```text
hs = [hs_key, hs_ref_1, hs_ref_2, ...]
```

5. Применяется temporal fusion:

```python
hs = self.temporal_query_fusion(hs, temporal_batch_size, temporal_num_frames)
```

После этого `hs` содержит только query key frame, но уже с учетом reference frames.

#### Baseline/inference

Если temporal выключен или на вход пришел одиночный кадр, используется обычный путь:

```text
samples -> backbone -> transformer -> heads
```

### Почему reference frames идут без градиентов

Не хватает памяти. Видеокарта geforce gtx 1070, 8GB.

Текущий вариант:

- key frame обучается полноценно;
- reference frames дают признаки для attention;
- temporal fusion обучается;
- расход памяти остается ближе к обычному RF-DETR.

Это компромисс между полноценным video training и доступной видеопамятью.

### Обработка `ref_unsigmoid`

После temporal fusion остаются только key-frame query. Поэтому `ref_unsigmoid` тоже приводится к key frame:

```python
ref_unsigmoid = ref_unsigmoid.view(... )[:, :, 0]
```

Это нужно, чтобы bbox head не получил reference anchors от соседних кадров.

### Segmentation compatibility

RF-DETR может иметь segmentation head. Хотя текущая задача - bbox detection, код оставляет совместимость. Если `features` содержат все кадры, перед segmentation head берется только key frame:

```python
segmentation_features = segmentation_features.view(... )[:, 0]
```

### Two-stage compatibility

RF-DETR может использовать two-stage encoder outputs. Для temporal режима encoder outputs тоже обрезаются до key frame:

```python
hs_enc = hs_enc.view(... )[:, 0]
ref_enc = ref_enc.view(... )[:, 0]
```

Это нужно, чтобы auxiliary encoder outputs соответствовали аннотациям key frame.

### Изменение `build_model`

В `LWDETR(...)` добавлена передача:

```python
temporal_num_ref_frames=getattr(args, "temporal_num_ref_frames", 0)
temporal_fusion_layers=getattr(args, "temporal_fusion_layers", 1)
temporal_dropout=getattr(args, "temporal_dropout", 0.0)
```

Это связывает CLI/config с моделью.

## Измененный файл: `external/rf-detr/src/rfdetr/datasets/coco.py`

В файл добавлен `TemporalCocoDetection`.

### Импорт `re`

Добавлен:

```python
import re
```

Он нужен, чтобы извлекать номер кадра из имени файла, если в COCO json нет `video_id`.

### `TemporalCocoDetection`

Новый класс наследуется от `CocoDetection`. Это важно: вся обычная логика RF-DETR для COCO сохраняется.

В `__getitem__` сначала загружается key frame обычным способом:

```python
key_img, target = super().__getitem__(idx)
```

Потом выбираются reference frames:

```python
for ref_id in self._reference_ids(key_image_id):
    ref_img = self._load_image_by_id(ref_id)
```

К каждому reference frame применяются те же transforms, что и к key frame:

```python
ref_img, _ = self._transforms(ref_img, None)
```

Итог:

```python
return torch.cat(frames, dim=0), target
```

Кадры склеиваются по канальному измерению.

### `_sequence_key`

Если `video_id` нет, метод пытается понять видео по имени файла. Например:

```text
cars11_frame_000120.jpg -> cars11_frame
```

Это запасной механизм для датасетов, где последовательность можно восстановить по имени.

### `_build_sequence_groups`

Группирует все изображения по видео:

```text
video_id -> [image_id_1, image_id_2, ...]
```

Потом сортирует кадры по имени файла. Это нужно, чтобы reference frame выбирался по порядку кадров.

### `_reference_ids`

Возвращает список `image_id` соседних кадров.

Для `previous`:

```text
[-step, -2*step, -3*step, ...]
```

Для `surrounding`:

```text
[-step, +step, -2*step, +2*step, ...]
```

Если индекс выходит за границы видео, он обрезается:

```python
ref_pos = min(max(pos + offset, 0), len(sequence) - 1)
```

Это защищает начало и конец видео.

### Изменение `build_coco`

Раньше всегда создавался `CocoDetection`. Теперь выбор зависит от `temporal_num_ref_frames`:

```python
dataset_cls = TemporalCocoDetection if temporal_num_ref_frames > 0 else CocoDetection
```

Если temporal выключен, baseline использует старый датасет.

Также добавлены:

```python
temporal_kwargs = {
    "num_ref_frames": temporal_num_ref_frames,
    "ref_frame_mode": ...,
    "ref_frame_step": ...,
}
```

Они передаются только в temporal dataset.

### Почему для temporal train используется deterministic transform

В коде:

```python
transform_image_set = "val" if temporal_num_ref_frames > 0 and image_set == "train" else image_set
```

Причина: key frame и reference frames должны иметь согласованную геометрию. Если применить разные random augmentations к соседним кадрам, temporal attention будет смотреть на несогласованные изображения.

Это не подходит для честного сравнения, потому что baseline может использовать train augmentations, а temporal - deterministic resize/normalize. Этот момент нужно учитывать при анализе результатов.

### Изменение `build_roboflow_from_coco`

Аналогичные изменения добавлены и в Roboflow COCO path:

- выбор `TemporalCocoDetection`;
- передача `temporal_kwargs`;
- deterministic transforms для temporal train.

Это сделано для совместимости с разными входными форматами RF-DETR.

## Измененный файл: `external/rf-detr/src/rfdetr/config.py`

Добавлены параметры в `ModelConfig`:

```python
temporal_num_ref_frames
temporal_fusion_layers
temporal_dropout
```

Назначение:

- `temporal_num_ref_frames` - включает temporal режим и задает число reference frames;
- `temporal_fusion_layers` - задает глубину temporal fusion;
- `temporal_dropout` - задает dropout внутри temporal attention.

Добавлены параметры в `TrainConfig`:

```python
temporal_ref_frame_mode
temporal_ref_frame_step
```

Назначение:

- `temporal_ref_frame_mode` - стратегия выбора reference frames;
- `temporal_ref_frame_step` - расстояние между кадрами.

Эти параметры нужны именно в train config, потому что выбор соседних кадров относится к датасету и training pipeline.

## Измененный файл: `external/rf-detr/src/rfdetr/_namespace.py`

RF-DETR использует внутренний namespace для передачи параметров из config в builder functions.

В `_MC_NAMESPACE_FIELDS` добавлены:

```python
temporal_dropout
temporal_fusion_layers
temporal_num_ref_frames
```

Без этого параметры остались бы в `ModelConfig`, но не дошли бы до `build_model`.

## Измененный файл: `external/rf-detr/src/rfdetr/models/_types.py`

В `BuilderArgs` добавлены:

```python
temporal_num_ref_frames: int
temporal_fusion_layers: int
temporal_dropout: float
```

Этот файл описывает, какие поля ожидают builder-функции модели. Изменение нужно для согласованности типов и документации внутренних аргументов RF-DETR.

## Новый файл: `external/rf-detr/tools/smoke_temporal_rfdetr.py`

Легкий smoke-test для temporal части. Он не обучает модель и не скачивает веса. Проверяет, что:

- `TemporalQueryFusion` принимает ожидаемые формы;
- `LWDETR` может пройти через temporal path;
- packed clip корректно разворачивается;
- выход модели остается совместимым с RF-DETR heads.

Этот файл полезен для быстрой проверки после изменений архитектуры.

## Команды

Команды вынесены в отдельные файлы:

```text
commands/train_baseline.sh
commands/train_temporal.sh
commands/train_baseline_then_temporal.sh
```

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

## Baseline command

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

## Temporal command

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

## Параметры обучения

```text
--variant small
```

Используется RF-DETR small. Это компромисс между качеством и памятью.

```text
--num-classes 1
```

В датасете один foreground class: автомобиль.

```text
--resolution 640
```

Размер, к которому RF-DETR приводит изображение во время обучения. Это не означает, что исходные изображения обязаны быть 640x640.

```text
--batch-size 1
```

Физический batch size. Для temporal режима память выше, поэтому batch size оставлен 1.

```text
--grad-accum-steps 4
```

Накопление градиента. Эффективный batch становится больше без роста памяти на один forward.

```text
--lr 1e-4
--lr-encoder 1e-5
```

Разные learning rate для основной модели и encoder/backbone части.

```text
--lr-drop 8
```

Learning rate scheduler снижает lr после 8 эпох в 10-epoch эксперименте.

## Параметры temporal

```text
--temporal-num-ref-frames 3
```

К каждому key frame добавляются 3 reference frames.

```text
--temporal-fusion-layers 1
```

Один temporal attention layer. Больше слоев могут дать больше выразительности, но увеличат память и время обучения.

```text
--temporal-ref-frame-mode surrounding
```

Берутся кадры вокруг текущего. Это offline режим. Для реального online/autopilot inference лучше использовать `previous`.

```text
--temporal-ref-frame-step 2
```

Reference frames берутся с шагом 2. Это уменьшает дублирование почти одинаковых соседних кадров.

## Результаты 10 эпох

| model | mAP 50:95 | EMA mAP 50:95 | AP50 | AP75 | mAR | val loss |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 0.7523 | 0.7540 | 0.9363 | 0.8290 | 0.8052 | 2.9813 |
| temporal | 0.7612 | 0.7621 | 0.9430 | 0.8407 | 0.8025 | 2.8761 |

Файлы результатов:

```text
results_10ep/baseline/metrics.csv
results_10ep/baseline/training_config.json
results_10ep/temporal/metrics.csv
results_10ep/temporal/training_config.json
results_10ep/logs/
reports/rfdetr_full_10ep_comparison/
```

По этим 10 эпохам temporal модель дала небольшой прирост по:

- `mAP 50:95`;
- `EMA mAP 50:95`;
- `AP50`;
- `AP75`;
- `val loss`.

Baseline немного выше по `mAR`. Это значит, что temporal лучше по точности/локализации, но recall нужно проверять отдельно.
