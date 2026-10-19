"""numpy-only logic shared by the precompute script and the Streamlit demo app"""
import numpy as np

GRID, WIN = 5, 24          # a 128x128 PASTIS patch is cut into a 5x5 grid of 24x24 windows (data2windows.py)
PIXEL_HA = 0.01            # Sentinel-2 10 m pixel = 100 m2 = 0.01 ha
RGB_IDX = [2, 1, 0]        # B4, B3, B2 in PASTIS band order

# PASTIS semantic classes 0-18 (19 = void, masked out); source: github.com/VSainteuf/pastis-benchmark
CLASS_NAMES = ["Background", "Meadow", "Soft winter wheat", "Corn", "Winter barley", "Winter rapeseed",
               "Spring barley", "Sunflower", "Grapevine", "Beet", "Winter triticale", "Winter durum wheat",
               "Fruits, vegetables, flowers", "Potatoes", "Leguminous fodder", "Soybeans", "Orchard",
               "Mixed cereal", "Sorghum"]
PALETTE = np.array([
    [40, 40, 40], [140, 200, 90], [255, 230, 0], [120, 210, 70], [230, 160, 40], [250, 250, 120],
    [200, 120, 30], [255, 180, 0], [100, 70, 150], [220, 80, 120], [180, 140, 60], [240, 200, 60],
    [255, 120, 120], [160, 100, 60], [80, 160, 120], [60, 120, 200], [30, 110, 50],
    [190, 170, 120], [150, 60, 40]], np.uint8)


def stitch(windows):
    """(25, ..., 24, 24) windows in row-major grid order -> (..., 120, 120)"""
    w = np.asarray(windows)
    lead = w.shape[1:-2]
    w = w.reshape((GRID, GRID) + lead + (WIN, WIN))
    w = np.moveaxis(w, (0, 1), (-4, -2))          # (..., GRID, WIN, GRID, WIN)
    return w.reshape(lead + (GRID * WIN, GRID * WIN))


def clip_roi(roi, shape):
    """(x, y, w, h) clipped to an image of shape (H, W, ...) -> (x0, y0, x1, y1)"""
    x, y, w, h = roi
    H, W = shape[:2]
    return max(0, x), max(0, y), min(W, x + w), min(H, y + h)


def roi_from_geojson(feature, size=GRID * WIN):
    """rectangle drawn on a Leaflet CRS.Simple map (lng = column, lat = size - row) -> (x, y, w, h) in pixels, or None"""
    if not feature:
        return None
    pts = np.asarray(feature['geometry']['coordinates'][0], dtype=float)
    x0, x1 = np.clip([np.floor(pts[:, 0].min()), np.ceil(pts[:, 0].max())], 0, size).astype(int)
    y0, y1 = np.clip([np.floor(size - pts[:, 1].max()), np.ceil(size - pts[:, 1].min())], 0, size).astype(int)
    if x1 <= x0 or y1 <= y0:
        return None
    return int(x0), int(y0), int(x1 - x0), int(y1 - y0)


def area_table(pred, roi):
    """crop rows (class_id, name, ha) inside roi sorted by area, total ha, background ha"""
    x0, y0, x1, y1 = clip_roi(roi, pred.shape)
    counts = np.bincount(pred[y0:y1, x0:x1].ravel(), minlength=len(CLASS_NAMES))
    rows = [(c, CLASS_NAMES[c], counts[c] * PIXEL_HA) for c in range(1, len(CLASS_NAMES)) if counts[c]]
    rows.sort(key=lambda r: -r[2])
    return rows, counts.sum() * PIXEL_HA, counts[0] * PIXEL_HA


def overlay(rgb, pred, roi, alpha=0.75):
    """class colors blended over rgb inside roi, with a white frame on the roi border"""
    out = rgb.copy()
    x0, y0, x1, y1 = clip_roi(roi, rgb.shape)
    if x1 <= x0 or y1 <= y0:
        return out
    region = out[y0:y1, x0:x1].astype(np.float32)
    colors = PALETTE[pred[y0:y1, x0:x1]].astype(np.float32)
    out[y0:y1, x0:x1] = (alpha * colors + (1 - alpha) * region).astype(np.uint8)
    out[y0, x0:x1] = out[y1 - 1, x0:x1] = 255
    out[y0:y1, x0] = out[y0:y1, x1 - 1] = 255
    return out


def clearest_date(img):
    """index of the frame with the darkest median RGB (clouds are bright); img (T, C, H, W) reflectance"""
    med = np.median(img[:, RGB_IDX].reshape(len(img), -1), axis=1)
    med = np.where(med > 0, med, np.inf)          # ignore empty frames
    return int(np.argmin(med))


def to_rgb_uint8(frame):
    """(C, H, W) reflectance -> (H, W, 3) uint8 with a 2-98 percentile stretch, for display only"""
    rgb = np.moveaxis(frame[RGB_IDX], 0, -1).astype(np.float32)
    lo, hi = np.percentile(rgb, 2), np.percentile(rgb, 98)
    return (np.clip((rgb - lo) / max(hi - lo, 1e-6), 0, 1) * 255).astype(np.uint8)


if __name__ == "__main__":
    # self-check: python -m app.demo_core
    # stitching: window j holds the constant j, so the stitched map must be row-major blocks of 24x24
    wins = np.stack([np.full((24, 24), j, np.uint8) for j in range(25)])
    m = stitch(wins)
    assert m.shape == (120, 120)
    for j in range(25):
        r, c = divmod(j, 5)
        assert (m[24 * r:24 * (r + 1), 24 * c:24 * (c + 1)] == j).all(), j
    # leading dims are kept: (25, T, C, 24, 24) -> (T, C, 120, 120)
    assert stitch(np.zeros((25, 3, 2, 24, 24))).shape == (3, 2, 120, 120)

    # areas: 1 px = 0.01 ha, only inside the ROI, background listed separately, totals match the ROI
    pred = np.zeros((120, 120), np.uint8)
    pred[:, 60:] = 2          # soft winter wheat on the right half
    pred[100:, 60:] = 8       # grapevine in the bottom-right corner
    rows, total, background = area_table(pred, (50, 90, 30, 30))   # x, y, w, h
    assert abs(total - 9.0) < 1e-9
    assert [r[0] for r in rows] == [8, 2] and abs(rows[0][2] - 4.0) < 1e-9 and abs(rows[1][2] - 2.0) < 1e-9, rows
    assert abs(background - 3.0) < 1e-9
    assert abs(sum(r[2] for r in rows) + background - total) < 1e-9
    # ROI past the border is clipped, not an error
    rows, total, _ = area_table(pred, (110, 110, 50, 50))
    assert abs(total - 1.0) < 1e-9 and rows[0][0] == 8

    # overlay keeps the image size and only changes pixels inside the ROI (plus its 1 px frame)
    rgb = np.zeros((120, 120, 3), np.uint8)
    out = overlay(rgb, pred, (60, 0, 10, 10))
    assert out.shape == rgb.shape and out.dtype == np.uint8
    assert (out[50:, :50] == 0).all() and out[5, 65].any()

    # clearest date: the darkest median RGB frame wins (clouds are bright)
    img = np.full((4, 10, 8, 8), 1000., np.float32)
    img[1] = 5000.
    img[2] = 300.
    assert clearest_date(img) == 2
    assert to_rgb_uint8(img[0]).shape == (8, 8, 3)

    # rectangle drawn on the map (Leaflet CRS.Simple: lng = column, lat = 120 - row) -> pixel roi
    def rect(x0, y0, x1, y1):
        return {'geometry': {'type': 'Polygon', 'coordinates': [[[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]]}}
    assert roi_from_geojson(rect(50, 0, 80, 30)) == (50, 90, 30, 30)
    assert roi_from_geojson(rect(50.4, 0.2, 79.6, 29.9)) == (50, 90, 30, 30)       # snapped outward to whole pixels
    assert roi_from_geojson(rect(-20, 100, 10, 160)) == (0, 0, 10, 20)               # clipped to the image
    assert roi_from_geojson(rect(130, 10, 150, 20)) is None                          # entirely outside
    assert roi_from_geojson(None) is None

    assert len(CLASS_NAMES) == 19 and PALETTE.shape == (19, 3)
    print("demo_core self-check passed")
