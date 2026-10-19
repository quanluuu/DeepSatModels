"""
Run TSViT_DINOv3 on whole PASTIS24 test patches and save what the demo app shows.
Run from the repo root (on Kaggle):
    python -m app.precompute --config configs/PASTIS24/TSViT_DINOv3_fold1.yaml --checkpoint <best.pth> --out app/data/demo_patches.npz
"""
import argparse
import os
import pickle
import re
from collections import defaultdict
import numpy as np
import pandas as pd
import torch
import data
from data.PASTIS24.data_transforms import PASTIS_segmentation_transform
from models import get_model
from utils.config_files_utils import read_yaml
from utils.torch_utils import load_from_checkpoint
from app.demo_core import GRID, stitch, clearest_date, to_rgb_uint8

VOID = 19


def complete_patches(paths):
    """{patch_id: [path of window 0..24]} for patches whose 25 windows are all listed"""
    windows = defaultdict(dict)
    for p in paths:
        m = re.search(r'(\d+)_(\d+)\.pickle$', p)
        windows[int(m.group(1))][int(m.group(2))] = p
    n = GRID * GRID
    return {pid: [w[j] for j in range(n)] for pid, w in windows.items() if all(j in w for j in range(n))}


def border_jump(img):
    """mean abs difference across the internal window borders of a stitched (C, H, W) image"""
    edges = np.arange(1, GRID) * (img.shape[-1] // GRID)
    return np.abs(img[..., edges] - img[..., edges - 1]).mean() + np.abs(img[..., edges, :] - img[..., edges - 1, :]).mean()


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True, help='trained TSViT_DINOv3 checkpoint')
    parser.add_argument('--num-patches', type=int, default=25)
    parser.add_argument('--candidates', type=int, default=100, help='complete test patches scanned to pick the demo set')
    parser.add_argument('--out', default='app/data/demo_patches.npz')
    args = parser.parse_args(argv)

    cfg = read_yaml(args.config)
    info = data.DATASET_INFO[cfg['DATASETS']['test']['dataset']]
    paths = pd.read_csv(info['paths_test'], header=None)[0].tolist()
    patches = complete_patches(paths)
    print('%d complete test patches' % len(patches))

    def load(pid):
        return [pickle.load(open(os.path.join(info['basedir'], p), 'rb'), encoding='latin1') for p in patches[pid]]

    def stitched_label(samples):
        return stitch(np.stack([s['labels'][0] for s in samples])).astype(np.uint8)

    # rank candidates by number of crop classes so the demo shows varied fields; keep only labels in memory
    cands = []
    for pid in sorted(patches)[:args.candidates]:
        n_crops = len(set(np.unique(stitched_label(load(pid)))) - {0, VOID})
        cands.append((-n_crops, pid))
    chosen = sorted(cands)[:args.num_patches]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = get_model(cfg, device)
    load_from_checkpoint(net, args.checkpoint, partial_restore=False, device=device)
    net.eval()
    transform = PASTIS_segmentation_transform(cfg['MODEL'], is_training=False)

    out = defaultdict(list)
    jumps = []
    for neg_crops, pid in chosen:
        samples = load(pid)
        label = stitched_label(samples)
        inputs = torch.stack([transform(dict(s))['inputs'] for s in samples]).to(device)   # (25, T, 24, 24, C)
        with torch.no_grad():
            pred = net(inputs).argmax(1).cpu().numpy().astype(np.uint8)                 # (25, 24, 24)
        img = stitch(np.stack([s['img'] for s in samples]).astype(np.float32))           # (T, C, 120, 120)
        t = clearest_date(img)
        # window order check: row-major stitching should give smaller seams than column-major
        wins = np.stack([s['img'][t] for s in samples]).astype(np.float32)
        col_major = wins.reshape((GRID, GRID) + wins.shape[1:]).swapaxes(0, 1).reshape(wins.shape)
        jumps.append((border_jump(stitch(wins)), border_jump(stitch(col_major))))

        out['patch_ids'].append(pid)
        out['rgb'].append(to_rgb_uint8(img[t]))
        out['pred'].append(stitch(pred))
        out['label'].append(np.where(label == VOID, 0, label))   # void pixels shown as background
        out['n_dates'].append(len(samples[0]['doy']))
        known = label != VOID
        print('patch %d: %d crop classes, pixel acc %.3f' % (pid, -neg_crops, (stitch(pred)[known] == label[known]).mean()))

    row, col = np.mean(jumps, axis=0)
    print('mean seam: row-major %.1f vs column-major %.1f (row-major should be smaller)' % (row, col))
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    np.savez_compressed(args.out, **{k: np.stack(v) for k, v in out.items()})
    print('saved %s (%.2f MB)' % (args.out, os.path.getsize(args.out) / 2**20))


if __name__ == "__main__":
    main()
