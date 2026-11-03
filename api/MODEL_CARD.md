# Model card: TSViT_DINOv3 inference API

| Field | Value |
| --- | --- |
| model_name | `TSViT_DINOv3` (`models/TSViT/TSViT_dinov3.py`), frozen DINOv3 ViT-S/16 per RGB frame + TSViT temporal and spatial encoders |
| model_version | `tsvit-dinov3-fold1-t9` (env `MODEL_VERSION`); returned in every `/health` and `/predict` response |
| checkpoint | `best_tsvit_dinov3.pth` from run T9 (30 epochs, warm start), 89 MB; holds the head and the DINOv3 backbone weights, so no separate DINOv3 weights file is needed (the backbone is built with `pretrained=False`) |
| trained_on | PASTIS24 fold 1: train folds 1-3, eval fold 4; one season (Sep 2018 - Nov 2019), France |
| metrics | Test split (fold 5): mIoU 53.9%, OA 79.6%. CPU latency for one window (local, 10 threads, n=10): p50 2.3 s, p95 2.4 s; peak RAM 889 MB |
| threshold | `CONFIDENCE_THRESHOLD` = 0.80 on the mean max-softmax of the window: at or above gives `OK`, below gives `UNCERTAIN` (taken from the course slides, not tuned on the eval split) |
| timeout | `INFER_TIMEOUT_S` = 10 s; a timeout or model error gives `FALLBACK` with `pred = null` |

## Input and output

- Input: one `.npz` file with `img` (T, 10, 24, 24), Sentinel-2 bands in PASTIS order, raw reflectance, and `doy` (T,), day of year 1..366. T is 1..60. Max file size is 5 MB. The file is read with `allow_pickle=False`.
- Preprocessing: `PASTIS_segmentation_transform(is_training=False)`, the same code as training: per-band fold 1 normalization, day-of-year channel, padding to 60 dates.
- Output: `pred` 24 x 24 class ids 0..18 (`class_names` gives the names), `confidence`, `status`, `area_ha` per predicted class (1 pixel = 0.01 ha), `model_version`, `latency_ms`.

## Known limitations

- Overfitting during training: eval loss rises from 0.577 at step 4000 to 0.965 at step 22000. The config uses no weight decay and no augmentation.
- DINOv3 sees only 3 of the 10 bands (B4, B3, B2) and was pretrained on web images.
- 4 x 4 pixel patches give coarser field boundaries than the original TSViT.
- One fold, one season, one country. Other regions and years are untested.
- A window is 240 m x 240 m; the API does not stitch larger areas.
- Confidence is the softmax of an uncalibrated model. A high value does not guarantee a correct map.

## Licenses

- Code: see `LICENSE.txt`.
- DINOv3 weights: DINOv3 License (Meta), see `facebookresearch/dinov3`.
- PASTIS data: CC BY 4.0, Sainte Fare Garnot & Landrieu (2021), https://zenodo.org/records/5012942.
