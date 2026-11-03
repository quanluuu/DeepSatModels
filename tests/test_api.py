"""inference API contract (SPEC-inference-api AC-AI-01..07) with a stub predictor: no torch weights needed"""
import io
import os
import time
import numpy as np
import pytest
from fastapi.testclient import TestClient
from api.main import create_app, status_of

FIXTURE = os.path.join(os.path.dirname(__file__), 'fixtures', 'window.npz')


class StubPredictor:
    model_version = 'stub-v0'
    device = 'cpu'

    def __init__(self, confidence=0.94, sleep=0., error=None):
        self.confidence, self.sleep, self.error = confidence, sleep, error
        self.calls = 0

    def predict(self, img, doy):
        self.calls += 1
        if self.error:
            raise self.error
        time.sleep(self.sleep)
        pred = np.zeros((24, 24), np.int64)
        pred[:, 12:] = 2        # soft winter wheat on the right half
        return pred, self.confidence


def client(stub, **kw):
    return TestClient(create_app(stub, **kw))


def npz_bytes(**arrays):
    buf = io.BytesIO()
    np.savez(buf, **arrays)
    return buf.getvalue()


def window(T=5):
    return dict(img=np.zeros((T, 10, 24, 24), np.int16), doy=np.arange(1, T + 1))


def post(c, raw, name='window.npz'):
    return c.post('/predict', files={'file': (name, raw, 'application/octet-stream')})


# AC-AI-01: threshold branches (white-box)
@pytest.mark.parametrize('confidence, status', [(0.95, 'OK'), (0.80, 'OK'), (0.79, 'UNCERTAIN'), (0.20, 'UNCERTAIN')])
def test_status_of(confidence, status):
    assert status_of(confidence, 0.80) == status


def test_health_reports_model_version():
    r = client(StubPredictor()).get('/health')
    assert r.status_code == 200
    assert r.json() == {'status': 'ok', 'model_version': 'stub-v0', 'device': 'cpu'}


# AC-AI-02
def test_confident_prediction_is_ok():
    with open(FIXTURE, 'rb') as f:
        r = post(client(StubPredictor(0.94)), f.read())
    body = r.json()
    assert r.status_code == 200 and body['status'] == 'OK'
    assert body['model_version'] == 'stub-v0' and body['latency_ms'] >= 0
    assert np.array(body['pred']).shape == (24, 24)
    assert body['confidence'] == pytest.approx(0.94)
    assert body['area_ha'] == {'Background': pytest.approx(2.88), 'Soft winter wheat': pytest.approx(2.88)}   # 288 px x 0.01 ha
    assert len(body['class_names']) == 19


# AC-AI-03
def test_low_confidence_is_uncertain():
    body = post(client(StubPredictor(0.55)), npz_bytes(**window())).json()
    assert body['status'] == 'UNCERTAIN' and body['pred'] is not None


def test_threshold_is_configurable():
    body = post(client(StubPredictor(0.55), threshold=0.5), npz_bytes(**window())).json()
    assert body['status'] == 'OK'


# AC-AI-04
def test_timeout_falls_back():
    body = post(client(StubPredictor(sleep=0.5), timeout_s=0.05), npz_bytes(**window())).json()
    assert body['status'] == 'FALLBACK' and body['pred'] is None and body['message']
    assert body['model_version'] == 'stub-v0'


def test_model_error_falls_back():
    body = post(client(StubPredictor(error=RuntimeError('boom'))), npz_bytes(**window())).json()
    assert body['status'] == 'FALLBACK' and body['pred'] is None


# AC-AI-05..07: bad input is rejected with 422 before the model is called
@pytest.mark.parametrize('raw', [
    b'MZ\x90\x00 not a zip',                                                     # .exe renamed to .npz
    npz_bytes(img=np.array([{'a': 1}], dtype=object), doy=np.arange(1)),        # object array (pickle)
    npz_bytes(img=np.zeros((5, 10, 24, 24))),                                    # missing doy
    npz_bytes(img=np.zeros((5, 10, 24, 23)), doy=np.arange(1, 6)),               # wrong window size
    npz_bytes(img=np.zeros((5, 9, 24, 24)), doy=np.arange(1, 6)),                # wrong band count
    npz_bytes(img=np.zeros((5, 10, 24, 24)), doy=np.arange(1, 5)),               # doy length != T
    npz_bytes(**window(61)),                                                     # more than 60 dates
    npz_bytes(img=np.zeros((0, 10, 24, 24)), doy=np.zeros(0)),                   # no dates
    npz_bytes(img=np.zeros((5, 10, 24, 24)), doy=np.array([0, 1, 2, 3, 400])),   # day of year out of range
    b'\x00' * (5 * 2**20 + 1),                                                   # larger than 5 MB
])
def test_bad_input_is_422_and_model_not_called(raw):
    stub = StubPredictor()
    r = post(client(stub), raw)
    assert r.status_code == 422 and r.json()['detail']
    assert stub.calls == 0


def test_plain_npy_is_rejected():
    buf = io.BytesIO()
    np.save(buf, np.zeros((5, 10, 24, 24)))
    stub = StubPredictor()
    assert post(client(stub), buf.getvalue()).status_code == 422 and stub.calls == 0
