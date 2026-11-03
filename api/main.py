"""
Inference API for TSViT_DINOv3: one 24x24 PASTIS window (.npz with img (T, 10, 24, 24) and doy (T,)) per request.
    uvicorn --factory api.main:create_app --host 0.0.0.0 --port 8000
Settings come from the environment, see .env.example.
"""
import io
import os
import time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from app.demo_core import CLASS_NAMES, PIXEL_HA

MAX_BYTES = 5 * 2**20       # 24x24 window, 60 dates, 10 bands as int16 is ~0.7 MB uncompressed
MAX_DATES = 60              # model max_seq_len
BANDS, WIN = 10, 24


def status_of(confidence, threshold):
    """OK when the mean max-softmax reaches the threshold, else UNCERTAIN (the user must review the map)"""
    return 'OK' if confidence >= threshold else 'UNCERTAIN'


def read_window(raw):
    """(img, doy) from the uploaded .npz bytes; ValueError with a user-facing message when the file is not a valid window"""
    if len(raw) > MAX_BYTES:
        raise ValueError('file larger than %d MB' % (MAX_BYTES // 2**20))
    try:
        d = np.load(io.BytesIO(raw), allow_pickle=False)    # never unpickle uploads: pickle can run arbitrary code
    except Exception:
        raise ValueError('not a valid .npz file')
    if not isinstance(d, np.lib.npyio.NpzFile):
        raise ValueError('expected a .npz archive with arrays img and doy')
    with d:
        missing = {'img', 'doy'} - set(d.files)
        if missing:
            raise ValueError('missing array(s): %s' % ', '.join(sorted(missing)))
        try:
            img, doy = d['img'], d['doy']
        except ValueError:
            raise ValueError('object arrays are not accepted')
    if not (np.issubdtype(img.dtype, np.number) and np.issubdtype(doy.dtype, np.integer)):
        raise ValueError('img must be numeric and doy integer')
    if img.ndim != 4 or img.shape[1:] != (BANDS, WIN, WIN):
        raise ValueError('img must have shape (T, %d, %d, %d), got %s' % (BANDS, WIN, WIN, img.shape))
    T = img.shape[0]
    if not 1 <= T <= MAX_DATES:
        raise ValueError('number of dates T must be 1..%d, got %d' % (MAX_DATES, T))
    if doy.shape != (T,):
        raise ValueError('doy must have shape (%d,), got %s' % (T, doy.shape))
    if doy.min() < 1 or doy.max() > 366:
        raise ValueError('doy values must be in 1..366')
    return img, doy


def create_app(predictor=None, threshold=None, timeout_s=None):
    """FastAPI app around a predictor with .predict(img, doy) -> (pred (24, 24), confidence), .model_version, .device;
    without a predictor the real model is loaded from the environment"""
    if predictor is None:
        from api.predictor import Predictor     # torch import only when serving the real model
        predictor = Predictor.from_env()
    threshold = float(os.environ.get('CONFIDENCE_THRESHOLD', 0.80)) if threshold is None else threshold
    timeout_s = float(os.environ.get('INFER_TIMEOUT_S', 10)) if timeout_s is None else timeout_s
    pool = ThreadPoolExecutor(max_workers=1)    # one inference at a time; a timed-out one still finishes in the background
    app = FastAPI(title='DeepSatModels inference API')

    @app.get('/health')
    def health():
        return {'status': 'ok', 'model_version': predictor.model_version, 'device': predictor.device}

    @app.post('/predict')
    def predict(file: UploadFile = File(...)):
        start = time.perf_counter()
        try:
            img, doy = read_window(file.file.read(MAX_BYTES + 1))
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))
        out = {'model_version': predictor.model_version, 'class_names': CLASS_NAMES}
        try:
            pred, confidence = pool.submit(predictor.predict, img, doy).result(timeout=timeout_s)
        except Exception as e:      # timeout or model failure: the caller falls back to manual work, the service stays up
            reason = 'timed out after %g s' % timeout_s if isinstance(e, TimeoutError) else 'failed (%s)' % type(e).__name__
            out.update(status='FALLBACK', confidence=None, pred=None, area_ha=None,
                       message='Inference %s; use the precomputed demo or label the area manually.' % reason)
        else:
            counts = np.bincount(np.asarray(pred).ravel(), minlength=len(CLASS_NAMES))
            out.update(status=status_of(confidence, threshold), confidence=round(float(confidence), 4),
                       pred=np.asarray(pred).tolist(),
                       area_ha={CLASS_NAMES[c]: round(n * PIXEL_HA, 4) for c, n in enumerate(counts) if n})
        out['latency_ms'] = round((time.perf_counter() - start) * 1000)
        return out

    return app
