"""attacks on the inference API (SPEC-ai-attack-suite ATK-03..10) with a stub predictor: no torch weights needed, runs on CI"""
import io
import os
import pickle
import time
import tracemalloc
import zipfile
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pytest
from test_api import StubPredictor, client, npz_bytes, post


def good(T=5):
    """a window that passes every input rule: in-range, non-constant reflectance and non-decreasing doy"""
    return dict(img=np.random.default_rng(0).integers(0, 3000, (T, 10, 24, 24)).astype(np.int16), doy=np.arange(1, T + 1) * 10)


def rejected(raw, **kw):
    stub = StubPredictor()
    r = post(client(stub, **kw), raw)
    return r.status_code in (413, 422) and stub.calls == 0


def with_img(img):
    return npz_bytes(img=img, doy=np.arange(1, len(img) + 1))


# ATK-03
@pytest.mark.parametrize('img', [
    np.full((5, 10, 24, 24), 32767, np.int16),                  # int16 max everywhere
    np.full((5, 10, 24, 24), -32768, np.int16),                 # int16 min everywhere
    np.where(np.arange(5 * 10 * 24 * 24).reshape(5, 10, 24, 24) == 0, 30000, good()['img']).astype(np.int16),    # one pixel 30000
    np.zeros((5, 10, 24, 24), np.int16),                        # constant: no data
], ids=['int16-max', 'int16-min', 'one-pixel-30000', 'constant'])
def test_out_of_range_reflectance_is_rejected(img):
    assert rejected(with_img(img))


# ATK-04
@pytest.mark.parametrize('bad', [np.nan, np.inf], ids=['nan', 'inf'])
def test_nan_or_inf_is_rejected(bad):
    img = good()['img'].astype(np.float32)
    img[2, 3, 4, 5] = bad
    assert rejected(with_img(img))


def bomb(n_dates=20000):
    """~1 MB .npz whose img is float64 (n_dates, 10, 24, 24), ~920 MB once decompressed; streamed, never held in memory here"""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as z:
        with z.open('img.npy', 'w', force_zip64=True) as f:
            np.lib.format.write_array_header_1_0(f, {'descr': '<f8', 'fortran_order': False, 'shape': (n_dates, 10, 24, 24)})
            chunk = bytes(2**20)
            left = n_dates * 10 * 24 * 24 * 8
            while left:
                f.write(chunk[:min(left, len(chunk))])
                left -= min(left, len(chunk))
        with z.open('doy.npy', 'w') as f:
            np.save(f, np.arange(1, 6))
    return buf.getvalue()


# ATK-05
def test_decompression_bomb_is_rejected_without_inflating():
    raw = bomb()
    assert len(raw) < 5 * 2**20     # passes the upload size limit
    stub = StubPredictor()
    c = client(stub)
    tracemalloc.start()
    r = post(c, raw)
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    print('ATK-05: %.1f MB upload, peak allocation %.0f MB, status %d' % (len(raw) / 2**20, peak / 2**20, r.status_code))
    assert r.status_code == 422 and stub.calls == 0
    assert peak < 50 * 2**20


class Payload:
    """unpickling this writes a marker file: proof that uploaded pickle ran code"""
    def __init__(self, marker):
        self.marker = marker

    def __reduce__(self):
        return open, (self.marker, 'w')     # creating the file is enough to prove code execution


# ATK-06
def test_pickle_payload_is_never_unpickled(tmp_path):
    marker = str(tmp_path / 'pwned')
    for raw in (npz_bytes(img=np.array([Payload(marker)], dtype=object), doy=np.arange(1, 2)),     # object array inside .npz
                pickle.dumps(Payload(marker))):                                                 # raw pickle upload
        assert rejected(raw)
    assert not os.path.exists(marker)


# ATK-07
@pytest.mark.parametrize('raw', [
    npz_bytes(**good())[:2000],                     # truncated zip
    b'\x89PNG\r\n\x1a\n' + npz_bytes(**good()),     # PNG header in front of a valid .npz (polyglot)
], ids=['truncated-zip', 'png-polyglot'])
def test_forged_file_is_rejected(raw):
    assert rejected(raw)


# ATK-07
def test_filename_and_content_type_are_ignored():
    name = '../../atk07_written.npz'    # path traversal: the API must never use the client's filename
    r = client(StubPredictor()).post('/predict', files={'file': (name, npz_bytes(**good()), 'image/png')})
    assert r.status_code == 200 and r.json()['status'] == 'OK'
    assert not os.path.exists(os.path.join(os.getcwd(), name))


# ATK-08
def test_oversized_upload_is_refused_before_reading():
    stub = StubPredictor()
    r = post(client(stub), bytes(50 * 2**20))      # TestClient sends Content-Length
    assert r.status_code == 413 and stub.calls == 0


# ATK-09
def test_flood_gets_503_and_health_stays_up():
    stub = StubPredictor(sleep=0.5)
    c = client(stub, timeout_s=5)
    raw = npz_bytes(**good())
    with ThreadPoolExecutor(12) as pool:
        futures = [pool.submit(post, c, raw) for _ in range(12)]
        time.sleep(0.2)
        t = time.perf_counter()
        health = c.get('/health')
        health_s = time.perf_counter() - t
        results = [f.result() for f in futures]
    codes = [r.status_code for r in results]
    statuses = [r.json()['status'] for r in results if r.status_code == 200]
    print('ATK-09: codes %s, statuses %s, model calls %d, /health %.2f s' % (sorted(codes), sorted(statuses), stub.calls, health_s))
    assert health.status_code == 200 and health_s < 1
    assert 503 in codes and all(r.headers.get('retry-after') for r in results if r.status_code == 503)
    assert 'FALLBACK' not in statuses       # what is accepted is served; the overflow is refused at once instead of timing out


# ATK-10
@pytest.mark.parametrize('doy', [np.arange(5, 0, -1) * 10, np.array([30, 10, 50, 20, 40]), np.array([10, 20, 20, 20, 30])],
                         ids=['reversed', 'shuffled', 'same-day-3-times'])
def test_doy_manipulation_is_rejected(doy):
    assert rejected(npz_bytes(img=good()['img'], doy=doy))


# ATK-10: real PASTIS windows repeat a day at most twice, which must still pass
def test_same_day_twice_is_accepted():
    r = post(client(StubPredictor()), npz_bytes(img=good()['img'], doy=np.array([10, 20, 20, 30, 40])))
    assert r.status_code == 200
