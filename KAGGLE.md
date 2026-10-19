# Running TSViT_DINOv3 on Kaggle

Notebook settings: **Accelerator: GPU T4** (not P100), **Internet: on** (needed for `pip install`; requires a phone-verified account).

## Kaggle Datasets (upload once)

| Slug (your choice) | Content |
|---|---|
| `<dataset-code>` | `DeepSatModels_code.zip` made with `git archive --format=zip -o DeepSatModels_code.zip HEAD` (Kaggle unzips it) |
| `<dataset-pastis24>` | folder `PASTIS24/` with the 24×24 pickles and `fold-paths/*.csv`, as on Google Drive |
| `<dataset-dinov3>` | folder `dinov3/` (snapshot of `facebookresearch/dinov3`, commit `6876159`) and `dinov3_vits16_pretrain_lvd1689m-08c60483.pth` |
| `<dataset-tsvit-ckpt>` | `best.pth` from `TSViT_fold1_full_a100_opt/` |

Check the real paths with `!ls /kaggle/input/*` after attaching the datasets; adjust the cells below if Kaggle nested the folders differently.

## Notebook cells

```python
# 1. writable copy of the code (configs save checkpoints relative to the repo root), found wherever Kaggle mounted it
import glob, os, shutil
def find(pattern):
    hits = glob.glob('/kaggle/input/**/' + pattern, recursive=True)
    assert hits, 'not found under /kaggle/input: %s (dataset attached? zip extracted?)' % pattern
    return hits[0]
code = find('models/TSViT/TSViT_dinov3.py').rsplit('/models/', 1)[0]
shutil.copytree(code, '/kaggle/working/DeepSatModels', dirs_exist_ok=True)
%cd /kaggle/working/DeepSatModels
```

```bash
# 2. missing packages (keep Kaggle's preinstalled torch/torchvision)
!pip install -q einops timm torchmetrics omegaconf ftfy regex submitit termcolor
```

```python
# 3. write the real dataset paths into the placeholders
paths = {
    '/kaggle/input/<dataset-pastis24>/PASTIS24': find('fold-paths/folds_1_123_paths.csv').rsplit('/fold-paths/', 1)[0],
    # weights before repo: the repo placeholder is a prefix of the weights placeholder
    '/kaggle/input/<dataset-dinov3>/dinov3_vits16_pretrain_lvd1689m-08c60483.pth': find('dinov3_vits16_pretrain_lvd1689m-08c60483.pth'),
    '/kaggle/input/<dataset-dinov3>/dinov3': find('dinov3/hub/backbones.py').rsplit('/dinov3/hub/', 1)[0],
    '/kaggle/input/<dataset-tsvit-ckpt>/best.pth': find('best.pth'),
}
for f in ['data/datasets.yaml', 'configs/PASTIS24/TSViT_DINOv3_fold1.yaml']:
    s = open(f).read()
    for k, v in paths.items():
        s = s.replace(k, v)
    open(f, 'w').write(s)
    assert '/kaggle/input/<dataset-' not in s, f
for k, v in paths.items(): print(v)
```

```bash
# 4. sanity checks: GPU, self-check, one real batch
!python -c "import torch; print(torch.__version__, torch.cuda.get_device_name(0), torch.cuda.get_arch_list())"
!python -m models.TSViT.TSViT_dinov3
!python -c "from utils.config_files_utils import read_yaml; from data import get_dataloaders; c = read_yaml('configs/PASTIS24/TSViT_DINOv3_fold1.yaml'); b = next(iter(get_dataloaders(c)['train'])); print({k: tuple(v.shape) for k, v in b.items()})"
```

The self-check skips its warm-start case on Kaggle (the old checkpoint is not at `TSViT_fold1_full_a100_opt/`); that case was verified locally.

## Smoke benchmark (T8)

Times `STEPS` training steps (after 5 warm-up steps) with the warm-started model and extrapolates to a full epoch.
Run once with `GPUS = 1`, once with `GPUS = 2` if the notebook has 2×T4. If it runs out of memory, lower `BATCH`.

```python
import time, torch
from torch import nn
from utils.config_files_utils import read_yaml
from utils.torch_utils import load_from_checkpoint
from data import get_dataloaders, get_loss_data_input
from models import get_model
from metrics.loss_functions import get_loss

CONFIG, BATCH, STEPS, GPUS = 'configs/PASTIS24/TSViT_DINOv3_fold1.yaml', 48, 100, 1
cfg = read_yaml(CONFIG)
cfg['DATASETS']['train']['batch_size'] = BATCH
dev = torch.device('cuda')
loaders = get_dataloaders(cfg)
net = get_model(cfg, dev)
load_from_checkpoint(net, cfg['CHECKPOINT']['load_from_checkpoint'], partial_restore=True,
                     restore_prefixes=cfg['CHECKPOINT']['restore_prefixes'])
if GPUS > 1:
    net = nn.DataParallel(net, device_ids=list(range(GPUS)))
net.to(dev).train()
opt = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=1e-4)
loss_fn, loss_in = get_loss(cfg, dev, reduction='mean'), get_loss_data_input(cfg)

torch.cuda.reset_peak_memory_stats()
losses, t0 = [], None
for step, sample in enumerate(loaders['train']):
    if step == 5:  # skip warm-up (cuDNN autotune, worker start-up)
        torch.cuda.synchronize(); t0 = time.time()
    out = net(sample['inputs'].to(dev)).permute(0, 2, 3, 1)
    loss = loss_fn(out, loss_in(sample, dev))
    opt.zero_grad(); loss.backward(); opt.step()
    losses.append(loss.item())
    if step == 5 + STEPS - 1:
        break
torch.cuda.synchronize()
sec = (time.time() - t0) / STEPS
n_train, n_eval = len(loaders['train']), len(loaders['eval'])
print('GPUS %d  batch %d  s/step %.3f  peak VRAM %.2f GiB' % (GPUS, BATCH, sec, torch.cuda.max_memory_allocated() / 2**30))
print('train steps/epoch %d  ->  %.1f min/epoch (train only)' % (n_train, n_train * sec / 60))
print('eval batches %d (batch %d)' % (n_eval, cfg['DATASETS']['eval']['batch_size']))
print('loss mean first 10 steps %.3f  last 10 steps %.3f' % (sum(losses[:10]) / 10, sum(losses[-10:]) / 10))
```

## Demo app data

Attach the output of the training notebook (it contains `models/saved_models/PASTIS24/TSViT_DINOv3_fold1/best.pth`)
as an input, or upload the trained checkpoint as a dataset **renamed to `best_tsvit_dinov3.pth`**. A downloaded
checkpoint may arrive as `best.zip`: it is the `.pth` file itself (PyTorch checkpoints are zip archives), so rename it
and do not extract it; Kaggle would unpack an uploaded `.zip`. Run cells 1–3, then:

```python
hits = glob.glob('/kaggle/input/**/best_tsvit_dinov3.pth', recursive=True)
ckpt = hits[0] if hits else find('TSViT_DINOv3_fold1/best.pth')
!python -m app.precompute --config configs/PASTIS24/TSViT_DINOv3_fold1.yaml --checkpoint {ckpt} --num-patches 25 --out app/data/demo_patches.npz
```

If Kaggle already extracted an uploaded checkpoint (you see `.../best/data.pkl` instead of a `.pth`), zip the folder
back into a checkpoint and pass that path as `ckpt`:

```python
import zipfile
src = os.path.dirname(find('best/data.pkl'))          # the extracted checkpoint folder
ckpt = '/kaggle/working/best_tsvit_dinov3.pth'
with zipfile.ZipFile(ckpt, 'w', zipfile.ZIP_STORED, strict_timestamps=False) as z:
    for dp, _, fs in os.walk(src):
        for f in sorted(fs):
            p = os.path.join(dp, f)
            z.write(p, os.path.relpath(p, os.path.dirname(src)))
```

Check that the printed `row-major` seam is clearly smaller than `column-major`, then download
`/kaggle/working/DeepSatModels/app/data/demo_patches.npz` from the notebook output.

## Runs

```bash
# train (warm-start from the old checkpoint, as set in the config)
!python train_and_eval/segmentation_training_transf.py --config configs/PASTIS24/TSViT_DINOv3_fold1.yaml --device 0

# random init: empty load_from_checkpoint first
!sed -i 's#^  load_from_checkpoint: .*#  load_from_checkpoint:#' configs/PASTIS24/TSViT_DINOv3_fold1.yaml

# test split of a trained run
!python train_and_eval/segmentation_training_transf.py --config configs/PASTIS24/TSViT_DINOv3_fold1.yaml --device 0 --test_only models/saved_models/PASTIS24/TSViT_DINOv3_fold1/best.pth
```

Outputs land in `/kaggle/working/DeepSatModels/models/saved_models/...`; save the notebook version (Save & Run All) to keep them.
