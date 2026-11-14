"""adversarial attacks on the real model (SPEC-ai-attack-suite ATK-01..02): run locally from the repo root, skipped without weights or the eval set
    python -m pytest -q tests/test_attack_model.py -rA
eps is in normalized units, i.e. a fraction of each band's std (Normalize divides by the fold 1 std).
"""
import contextlib
import os
from unittest import mock
import numpy as np
import pytest

EVAL = 'tests/fixtures/eval_windows.npz'
N_WINDOWS, EPS, VOID, THRESHOLD = 20, 0.03, 19, 0.80
missing = [p for p in ('weights/best.pth', 'weights/dinov3/hubconf.py', EVAL) if not os.path.exists(p)]
pytestmark = pytest.mark.skipif(bool(missing), reason='needs %s (see SPEC-ai-test-suite Commands)' % ', '.join(missing))


@pytest.fixture(scope='module')
def predictor():
    from api.predictor import Predictor
    return Predictor.from_env()


@pytest.fixture(scope='module')
def cases(predictor):
    """(normalized inputs (1, 60, 24, 24, 11), label (1, 24, 24), mask of the attackable values) for the first N eval windows"""
    import torch
    out = []
    with np.load(EVAL) as d:
        for i in range(N_WINDOWS):
            img, doy, label = d['img_%d' % i], d['doy_%d' % i], d['label_%d' % i]
            x = predictor.transform({'img': img.astype(np.float32), 'doy': doy, 'labels': np.zeros((1, 24, 24), np.uint8)})['inputs'][None]
            mask = torch.zeros_like(x)
            mask[:, :min(len(img), 60), :, :, :10] = 1      # real dates and the 10 bands only: no doy channel, no zero padding
            out.append((x, torch.as_tensor(label.astype(np.int64))[None], mask))
    return out


def evaluate(predictor, x):
    import torch
    with torch.no_grad():
        prob = predictor.model(x).softmax(1)[0]
    conf, pred = prob.max(0)
    return pred.numpy(), float(conf.mean())


def attack(predictor, x0, label, mask, steps, alpha):
    """untargeted L-inf attack: FGSM when steps == 1 and alpha == EPS, PGD otherwise"""
    import torch
    x = x0.clone()
    # the frozen DINOv3 backbone runs under torch.no_grad; lift it so the gradient reaches the input
    with mock.patch('models.TSViT.TSViT_dinov3.torch.no_grad', contextlib.nullcontext):
        for _ in range(steps):
            x.requires_grad_(True)
            loss = torch.nn.functional.cross_entropy(predictor.model(x), label, ignore_index=VOID)
            grad, = torch.autograd.grad(loss, x)
            x = (x.detach() + alpha * grad.sign() * mask).clamp(x0 - EPS, x0 + EPS)
    return x.detach()


def run(predictor, cases, steps, alpha, record_property, name):
    import torch
    gen = torch.Generator().manual_seed(0)
    correct, total, uncertain, flipped = {'clean': 0, 'random': 0, 'attack': 0}, 0, 0, 0
    for x0, label, mask in cases:
        lab = label[0].numpy()
        keep = lab != VOID
        clean, _ = evaluate(predictor, x0)
        rand, _ = evaluate(predictor, x0 + EPS * torch.randn(x0.shape, generator=gen).sign() * mask)    # same L-inf budget, no gradient
        adv, conf = evaluate(predictor, attack(predictor, x0, label, mask, steps, alpha))
        for k, pred in (('clean', clean), ('random', rand), ('attack', adv)):
            correct[k] += (pred[keep] == lab[keep]).sum()
        total += keep.sum()
        uncertain += conf < THRESHOLD
        flipped += (adv != clean).mean()
    acc = {k: v / total for k, v in correct.items()}
    for k, v in acc.items():
        record_property('acc_' + k, round(float(v), 4))
    record_property('uncertain_rate', uncertain / len(cases))
    record_property('flip_rate', round(float(flipped / len(cases)), 4))
    print('%s eps %.2f std, %d windows: accuracy clean %.1f%%, random noise %.1f%%, attacked %.1f%%; pixels flipped %.1f%%; UNCERTAIN %d/%d'
          % (name, EPS, len(cases), 100 * acc['clean'], 100 * acc['random'], 100 * acc['attack'], 100 * flipped / len(cases), uncertain, len(cases)))
    # safe if the attack barely hurts, or if the API flags most attacked windows for review
    assert acc['clean'] - acc['attack'] <= 0.10 or uncertain / len(cases) >= 0.8


# known vulnerability, not fixed (needs adversarial training): strict, so a model that becomes robust turns this red and the mark gets removed
VULNERABLE = 'measured 2026-11-14 on 20 windows, eps 0.03 std: %s'


# ATK-01
@pytest.mark.xfail(strict=True, reason=VULNERABLE % 'accuracy 80.8% -> 57.1%, UNCERTAIN 1/20')
def test_fgsm(predictor, cases, record_property):
    run(predictor, cases, steps=1, alpha=EPS, record_property=record_property, name='ATK-01 FGSM')


# ATK-02
@pytest.mark.xfail(strict=True, reason=VULNERABLE % 'accuracy 80.8% -> 12.3%, UNCERTAIN 0/20')
def test_pgd(predictor, cases, record_property):
    run(predictor, cases, steps=10, alpha=0.01, record_property=record_property, name='ATK-02 PGD')
