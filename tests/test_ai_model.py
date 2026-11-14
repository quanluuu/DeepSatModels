"""real-model tests (SPEC-ai-test-suite TC-06..08) through the API: run locally from the repo root, skipped when weights or the eval set are missing
    python -m pytest -q tests/test_ai_model.py -rA --junitxml docs/report/qa/junit-ai.xml
"""
import io
import os
import numpy as np
import pytest
from fastapi.testclient import TestClient
from api.main import create_app

FIXTURE = 'tests/fixtures/window.npz'
EVAL = 'tests/fixtures/eval_windows.npz'
N_CLASSES, VOID = 19, 19
missing = [p for p in ('weights/best.pth', 'weights/dinov3/hubconf.py', EVAL) if not os.path.exists(p)]
pytestmark = pytest.mark.skipif(bool(missing), reason='needs %s (see SPEC-ai-test-suite Commands)' % ', '.join(missing))


@pytest.fixture(scope='module')
def api():
    return TestClient(create_app(timeout_s=120))    # measure the model, not CPU latency: a 10 s timeout would turn slow runs into FALLBACK


@pytest.fixture(scope='module')
def windows():
    with np.load(EVAL) as d:
        return [(d['img_%d' % i], d['doy_%d' % i], d['label_%d' % i]) for i in range(len(d.files) // 3)]


@pytest.fixture(scope='module')
def preds(api, windows):
    """API prediction for every eval window, shared by TC-06 and TC-08"""
    return [predict(api, img, doy) for img, doy, _ in windows]


def predict(api, img, doy):
    buf = io.BytesIO()
    np.savez(buf, img=img, doy=doy)
    body = api.post('/predict', files={'file': ('w.npz', buf.getvalue(), 'application/octet-stream')}).json()
    assert body['status'] in ('OK', 'UNCERTAIN'), body.get('message')
    return np.array(body['pred']), body['confidence']


# TC-06
def test_accuracy_on_fold5_windows(windows, preds, record_property):
    label = np.concatenate([w[2].ravel() for w in windows])
    pred = np.concatenate([p.ravel() for p, _ in preds])
    keep = label != VOID
    cm = np.bincount(N_CLASSES * label[keep].astype(np.int64) + pred[keep], minlength=N_CLASSES ** 2).reshape(N_CLASSES, N_CLASSES)
    tp, union = np.diag(cm), cm.sum(0) + cm.sum(1) - np.diag(cm)
    oa, miou = tp.sum() / cm.sum(), (tp[union > 0] / union[union > 0]).mean()     # classes present in label or prediction
    record_property('OA', round(float(oa), 4))
    record_property('mIoU', round(float(miou), 4))
    print('TC-06: %d windows, %d labelled pixels, OA %.1f%%, mIoU %.1f%% over %d classes'
          % (len(windows), keep.sum(), 100 * oa, 100 * miou, (union > 0).sum()))
    assert oa >= 0.746 and miou >= 0.489    # model card (OA 79.6%, mIoU 53.9%) minus 5 points


# TC-07
def test_same_input_gives_same_prediction(api):
    with np.load(FIXTURE) as d:
        img, doy = d['img'], d['doy']
    (p1, c1), (p2, c2) = predict(api, img, doy), predict(api, img, doy)
    print('TC-07: confidence %s and %s, %d pixels differ' % (c1, c2, (p1 != p2).sum()))
    assert (p1 == p2).all() and c1 == c2


# TC-08
def test_small_noise_keeps_prediction(api, windows, preds, record_property):
    rng = np.random.default_rng(0)
    agree = []
    for (img, doy, _), (base, _) in zip(windows[:10], preds[:10]):
        noisy = (img * rng.uniform(0.99, 1.01, img.shape)).astype(np.float32)    # +-1% reflectance per value
        agree.append((predict(api, noisy, doy)[0] == base).mean())
    record_property('agreement_mean', round(float(np.mean(agree)), 4))
    print('TC-08: pixel agreement mean %.1f%%, min %.1f%% over %d windows' % (100 * np.mean(agree), 100 * min(agree), len(agree)))
    assert np.mean(agree) >= 0.95
