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


def to_lonlat(geo, cols, rows):
    """pixel corner coordinates -> (lon, lat); geo is the (2, 3) affine [lon, lat] = geo @ [col, row, 1]"""
    return geo @ np.stack([cols, rows, np.ones_like(cols)])


def to_pixel(geo, lon, lat):
    """(lon, lat) -> fractional (col, row), inverse of to_lonlat"""
    return np.linalg.solve(geo[:, :2], np.stack([np.subtract(lon, geo[0, 2]), np.subtract(lat, geo[1, 2])]))


def outline(geo, roi):
    """[lat, lon] corners of a pixel roi (x, y, w, h), clockwise from the top-left, for a Leaflet polygon"""
    x, y, w, h = roi
    lon, lat = to_lonlat(geo, np.array([x, x + w, x + w, x], float), np.array([y, y, y + h, y + h], float))
    return np.stack([lat, lon], 1).tolist()


def geo_bounds(geo, size=GRID * WIN):
    """[[lat_min, lon_min], [lat_max, lon_max]] of a size x size patch"""
    pts = np.array(outline(geo, (0, 0, size, size)))
    return [pts.min(0).tolist(), pts.max(0).tolist()]


def roi_from_lonlat(geo, feature, size=GRID * WIN):
    """rectangle drawn on the map (GeoJSON lon/lat) -> pixel roi (x, y, w, h) of its bounding box, or None"""
    if not feature:
        return None
    lon, lat = np.asarray(feature['geometry']['coordinates'][0], dtype=float).T
    col, row = np.round(to_pixel(geo, lon, lat), 6)       # rounding keeps exact pixel edges from floor/ceil jitter
    x0, x1 = np.clip([np.floor(col.min()), np.ceil(col.max())], 0, size).astype(int)
    y0, y1 = np.clip([np.floor(row.min()), np.ceil(row.max())], 0, size).astype(int)
    if x1 <= x0 or y1 <= y0:
        return None
    return int(x0), int(y0), int(x1 - x0), int(y1 - y0)


def warp(img, geo, roi=None, out=256):
    """(H, W, 3) image resampled (nearest) onto the lon/lat grid of its bounding box, row 0 = north -> (out, out, 4)
    RGBA, transparent outside the image or the roi"""
    H, W = img.shape[:2]
    (lat0, lon0), (lat1, lon1) = geo_bounds(geo, H)
    t = (np.arange(out) + 0.5) / out
    lon, lat = np.meshgrid(lon0 + t * (lon1 - lon0), lat1 - t * (lat1 - lat0))
    col, row = np.floor(to_pixel(geo, lon.ravel(), lat.ravel())).astype(int).reshape(2, out, out)
    x0, y0, x1, y1 = clip_roi(roi or (0, 0, W, H), img.shape)
    rgba = np.zeros((out, out, 4), np.uint8)
    rgba[..., :3] = img[np.clip(row, 0, H - 1), np.clip(col, 0, W - 1)]
    rgba[..., 3] = 255 * ((col >= x0) & (col < x1) & (row >= y0) & (row < y1))
    return rgba


def area_table(pred, roi):
    """crop rows (class_id, name, ha) inside roi sorted by area, total ha, background ha"""
    x0, y0, x1, y1 = clip_roi(roi, pred.shape)
    counts = np.bincount(pred[y0:y1, x0:x1].ravel(), minlength=len(CLASS_NAMES))
    rows = [(c, CLASS_NAMES[c], counts[c] * PIXEL_HA) for c in range(1, len(CLASS_NAMES)) if counts[c]]
    rows.sort(key=lambda r: -r[2])
    return rows, counts.sum() * PIXEL_HA, counts[0] * PIXEL_HA


def panels(rgb, pred, label, roi, scale=4):
    """RGB, prediction and ground truth of the roi, each upscaled by repeating pixels"""
    x0, y0, x1, y1 = clip_roi(roi, rgb.shape)
    imgs = [rgb[y0:y1, x0:x1], PALETTE[pred[y0:y1, x0:x1]], PALETTE[label[y0:y1, x0:x1]]]
    return [i.repeat(scale, 0).repeat(scale, 1) for i in imgs]


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

    rgb = np.zeros((120, 120, 3), np.uint8)

    # clearest date: the darkest median RGB frame wins (clouds are bright)
    img = np.full((4, 10, 8, 8), 1000., np.float32)
    img[1] = 5000.
    img[2] = 300.
    assert clearest_date(img) == 2
    assert to_rgb_uint8(img[0]).shape == (8, 8, 3)

    # georeference: [lon, lat] = geo @ [col, row, 1]; north-up patch at 2 E, 48 N with 1e-4 degree pixels
    north = np.array([[1e-4, 0, 2.0], [0, -1e-4, 48.0]])
    turned = np.array([[0, -1e-4, 2.0], [-1e-4, 0, 48.0]])      # rotated 90 degrees: lon = 2 - row, lat = 48 - col
    assert np.allclose(geo_bounds(north), [[48 - 120e-4, 2], [48, 2 + 120e-4]])
    assert np.allclose(geo_bounds(turned), [[48 - 120e-4, 2 - 120e-4], [48, 2]])
    assert np.allclose(outline(north, (10, 20, 30, 40)), [[48 - 20e-4, 2 + 10e-4], [48 - 20e-4, 2 + 40e-4],
                                                         [48 - 60e-4, 2 + 40e-4], [48 - 60e-4, 2 + 10e-4]])

    # rectangle drawn on the map (GeoJSON lon/lat) -> pixel roi of its bounding box
    def rect(lon0, lat0, lon1, lat1):
        return {'geometry': {'type': 'Polygon', 'coordinates': [[[lon0, lat0], [lon1, lat0], [lon1, lat1], [lon0, lat1], [lon0, lat0]]]}}

    def px_rect(x0, y0, x1, y1):                    # same rectangle in pixels on the north-up patch
        return rect(2 + x0 * 1e-4, 48 - y1 * 1e-4, 2 + x1 * 1e-4, 48 - y0 * 1e-4)
    assert roi_from_lonlat(north, px_rect(50, 90, 80, 120)) == (50, 90, 30, 30)
    assert roi_from_lonlat(north, px_rect(50.4, 90.1, 79.6, 119.8)) == (50, 90, 30, 30)   # snapped outward to whole pixels
    assert roi_from_lonlat(north, px_rect(-20, -40, 10, 20)) == (0, 0, 10, 20)              # clipped to the image
    assert roi_from_lonlat(north, px_rect(130, 10, 150, 20)) is None                        # entirely outside
    assert roi_from_lonlat(north, None) is None
    assert roi_from_lonlat(turned, rect(2 - 30e-4, 48 - 60e-4, 2 - 10e-4, 48 - 50e-4)) == (50, 10, 10, 20)

    # warp onto the lon/lat grid of the bounding box (row 0 = north); alpha 255 only inside the roi
    img = np.random.default_rng(0).integers(0, 255, (120, 120, 3), dtype=np.uint8)
    out = warp(img, north, out=120)
    assert out.shape == (120, 120, 4) and out.dtype == np.uint8
    assert (out[..., :3] == img).all() and (out[..., 3] == 255).all()
    out = warp(img, north, roi=(60, 0, 10, 10), out=120)
    assert (out[:10, 60:70, 3] == 255).all() and out[..., 3].sum() == 100 * 255
    out = warp(img, turned, out=120)
    assert (out[0, 0, :3] == img[119, 0]).all() and (out[119, 119, :3] == img[0, 119]).all()

    # side-by-side panels: RGB, prediction, ground truth of the ROI, upscaled by whole pixels
    label = np.full_like(pred, 5)
    imgs = panels(rgb, pred, label, (55, 95, 10, 10), scale=3)
    assert [p.shape for p in imgs] == [(30, 30, 3)] * 3 and all(p.dtype == np.uint8 for p in imgs)
    assert (imgs[1][:, :15] == PALETTE[0]).all() and (imgs[1][:15, 15:] == PALETTE[2]).all()
    assert (imgs[1][15:, 15:] == PALETTE[8]).all() and (imgs[2] == PALETTE[5]).all()

    assert len(CLASS_NAMES) == 19 and PALETTE.shape == (19, 3)
    print("demo_core self-check passed")
