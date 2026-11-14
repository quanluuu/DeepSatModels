"""scripts/make_fixture.py --eval (SPEC-ai-test-suite T2) on a tiny fake PASTIS24 zip: no dataset needed"""
import pickle
import zipfile
import numpy as np
from scripts.make_fixture import ROOT, main


def fake_zip(path, n):
    paths = ['pickle24x24/%d_0.pickle' % i for i in range(n)]
    with zipfile.ZipFile(path, 'w') as z:
        z.writestr(ROOT + 'fold-paths/fold_5_paths.csv', '\n'.join(paths))
        for i, p in enumerate(paths):
            T = 3 + i       # T differs per window, so the arrays cannot be stacked
            z.writestr(ROOT + p, pickle.dumps({'img': np.full((T, 10, 24, 24), i, np.float64),
                                               'labels': np.stack([np.full((24, 24), i), np.zeros((24, 24)), np.ones((24, 24))]),
                                               'doy': np.arange(1, T + 1, dtype=np.float64)}))


def test_eval_picks_evenly_spaced_windows_with_labels(tmp_path):
    fake_zip(tmp_path / 'p.zip', 6)
    out = tmp_path / 'eval.npz'
    main(['--zip', str(tmp_path / 'p.zip'), '--eval', '3', '--out', str(out)])
    with np.load(out) as d:
        assert sorted(d.files) == ['doy_0', 'doy_1', 'doy_2', 'img_0', 'img_1', 'img_2', 'label_0', 'label_1', 'label_2']
        for i, src in enumerate([0, 2, 4]):     # every len/n-th path
            assert d['img_%d' % i].dtype == np.int16 and d['img_%d' % i].shape == (3 + src, 10, 24, 24)
            assert d['doy_%d' % i].dtype == np.int16 and d['doy_%d' % i].shape == (3 + src,)
            assert d['label_%d' % i].dtype == np.uint8 and (d['label_%d' % i] == src).all()     # labels[0] = semantic class
