"""
Cut one PASTIS24 test window into the .npz the inference API takes (img (T, 10, 24, 24), doy (T,)).
The window comes from a patch in app/data/demo_patches.npz, so the API prediction can be checked against the demo.
    python -I scripts/make_fixture.py --zip pastis24.zip --out tests/fixtures/window.npz
"""
import argparse
import os
import pickle
import re
import zipfile
import numpy as np

ROOT = 'PASTIS24_25parts/'


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--zip', required=True, help='PASTIS24 zip (PASTIS24_25parts/...)')
    parser.add_argument('--demo', default='app/data/demo_patches.npz')
    parser.add_argument('--out', default='tests/fixtures/window.npz')
    args = parser.parse_args(argv)

    with np.load(args.demo) as d:
        demo_ids = set(d['patch_ids'].tolist())
    with zipfile.ZipFile(args.zip) as z:
        paths = z.read(ROOT + 'fold-paths/fold_5_paths.csv').decode().split()     # fold 5 = test split of fold 1
        # first listed window of a demo patch; window 0 keeps the stitched position simple (top-left 24x24)
        path = next(p for p in paths if re.search(r'(\d+)_0\.pickle$', p) and int(re.search(r'(\d+)_0', p).group(1)) in demo_ids)
        sample = pickle.load(z.open(ROOT + path), encoding='latin1')    # trusted local dataset, never user uploads

    img, doy = sample['img'].astype(np.int16), sample['doy'].astype(np.int16)
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    np.savez_compressed(args.out, img=img, doy=doy)
    print('%s -> %s: img %s, doy %s, %.0f KB' % (path, args.out, img.shape, doy.shape, os.path.getsize(args.out) / 1024))


if __name__ == "__main__":
    main()
