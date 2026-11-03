"""unit tests for app/demo_core.py (ported from its __main__ self-check)"""
import numpy as np
import pytest
from demo_core import (CLASS_NAMES, PALETTE, area_table, clearest_date, clip_roi, geo_bounds, outline, panels, roi_from_lonlat,
                       stitch, to_lonlat, to_pixel, to_rgb_uint8, warp)

# [lon, lat] = geo @ [col, row, 1]; north-up patch at 2 E, 48 N with 1e-4 degree pixels
NORTH = np.array([[1e-4, 0, 2.0], [0, -1e-4, 48.0]])
TURNED = np.array([[0, -1e-4, 2.0], [-1e-4, 0, 48.0]])      # rotated 90 degrees: lon = 2 - row, lat = 48 - col


@pytest.fixture
def pred():
    p = np.zeros((120, 120), np.uint8)
    p[:, 60:] = 2          # soft winter wheat on the right half
    p[100:, 60:] = 8       # grapevine in the bottom-right corner
    return p


def rect(lon0, lat0, lon1, lat1):
    return {'geometry': {'type': 'Polygon', 'coordinates': [[[lon0, lat0], [lon1, lat0], [lon1, lat1], [lon0, lat1], [lon0, lat0]]]}}


def px_rect(x0, y0, x1, y1):                    # same rectangle in pixels on the north-up patch
    return rect(2 + x0 * 1e-4, 48 - y1 * 1e-4, 2 + x1 * 1e-4, 48 - y0 * 1e-4)


def test_stitch_is_row_major():
    # window j holds the constant j, so the stitched map must be row-major blocks of 24x24
    m = stitch(np.stack([np.full((24, 24), j, np.uint8) for j in range(25)]))
    assert m.shape == (120, 120)
    for j in range(25):
        r, c = divmod(j, 5)
        assert (m[24 * r:24 * (r + 1), 24 * c:24 * (c + 1)] == j).all(), j


def test_stitch_keeps_leading_dims():
    assert stitch(np.zeros((25, 3, 2, 24, 24))).shape == (3, 2, 120, 120)


def test_clip_roi_inside_and_past_border():
    assert clip_roi((10, 20, 30, 40), (120, 120)) == (10, 20, 40, 60)
    assert clip_roi((-5, 110, 20, 20), (120, 120, 3)) == (0, 110, 15, 120)


def test_area_table_counts_only_inside_roi(pred):
    # 1 px = 0.01 ha, background listed separately, totals match the ROI
    rows, total, background = area_table(pred, (50, 90, 30, 30))   # x, y, w, h
    assert abs(total - 9.0) < 1e-9
    assert [r[0] for r in rows] == [8, 2] and abs(rows[0][2] - 4.0) < 1e-9 and abs(rows[1][2] - 2.0) < 1e-9, rows
    assert abs(background - 3.0) < 1e-9
    assert abs(sum(r[2] for r in rows) + background - total) < 1e-9


def test_area_table_clips_roi_past_border(pred):
    rows, total, _ = area_table(pred, (110, 110, 50, 50))
    assert abs(total - 1.0) < 1e-9 and rows[0][0] == 8


def test_clearest_date_picks_darkest_frame():
    # clouds are bright, so the darkest median RGB frame wins
    img = np.full((4, 10, 8, 8), 1000., np.float32)
    img[1] = 5000.
    img[2] = 300.
    assert clearest_date(img) == 2


def test_to_rgb_uint8_shape_and_dtype():
    out = to_rgb_uint8(np.random.default_rng(0).uniform(0, 3000, (10, 8, 8)).astype(np.float32))
    assert out.shape == (8, 8, 3) and out.dtype == np.uint8


def test_to_pixel_inverts_to_lonlat():
    cols, rows = np.array([0., 10.5, 120.]), np.array([0., 3.25, 120.])
    for geo in (NORTH, TURNED):
        lon, lat = to_lonlat(geo, cols, rows)
        assert np.allclose(to_pixel(geo, lon, lat), [cols, rows])


def test_geo_bounds_and_outline():
    assert np.allclose(geo_bounds(NORTH), [[48 - 120e-4, 2], [48, 2 + 120e-4]])
    assert np.allclose(geo_bounds(TURNED), [[48 - 120e-4, 2 - 120e-4], [48, 2]])
    assert np.allclose(outline(NORTH, (10, 20, 30, 40)), [[48 - 20e-4, 2 + 10e-4], [48 - 20e-4, 2 + 40e-4],
                                                         [48 - 60e-4, 2 + 40e-4], [48 - 60e-4, 2 + 10e-4]])


@pytest.mark.parametrize('feature, roi', [
    (px_rect(50, 90, 80, 120), (50, 90, 30, 30)),
    (px_rect(50.4, 90.1, 79.6, 119.8), (50, 90, 30, 30)),   # snapped outward to whole pixels
    (px_rect(-20, -40, 10, 20), (0, 0, 10, 20)),             # clipped to the image
    (px_rect(130, 10, 150, 20), None),                       # entirely outside
    (None, None),
])
def test_roi_from_lonlat_north_up(feature, roi):
    assert roi_from_lonlat(NORTH, feature) == roi


def test_roi_from_lonlat_rotated():
    assert roi_from_lonlat(TURNED, rect(2 - 30e-4, 48 - 60e-4, 2 - 10e-4, 48 - 50e-4)) == (50, 10, 10, 20)


def test_warp_north_up_is_identity_with_full_alpha():
    img = np.random.default_rng(0).integers(0, 255, (120, 120, 3), dtype=np.uint8)
    out = warp(img, NORTH, out=120)
    assert out.shape == (120, 120, 4) and out.dtype == np.uint8
    assert (out[..., :3] == img).all() and (out[..., 3] == 255).all()


def test_warp_alpha_only_inside_roi():
    img = np.random.default_rng(0).integers(0, 255, (120, 120, 3), dtype=np.uint8)
    out = warp(img, NORTH, roi=(60, 0, 10, 10), out=120)
    assert (out[:10, 60:70, 3] == 255).all() and out[..., 3].sum() == 100 * 255


def test_warp_rotated_puts_row0_north():
    img = np.random.default_rng(0).integers(0, 255, (120, 120, 3), dtype=np.uint8)
    out = warp(img, TURNED, out=120)
    assert (out[0, 0, :3] == img[119, 0]).all() and (out[119, 119, :3] == img[0, 119]).all()


def test_panels_rgb_pred_label_upscaled(pred):
    rgb = np.zeros((120, 120, 3), np.uint8)
    label = np.full_like(pred, 5)
    imgs = panels(rgb, pred, label, (55, 95, 10, 10), scale=3)
    assert [p.shape for p in imgs] == [(30, 30, 3)] * 3 and all(p.dtype == np.uint8 for p in imgs)
    assert (imgs[1][:, :15] == PALETTE[0]).all() and (imgs[1][:15, 15:] == PALETTE[2]).all()
    assert (imgs[1][15:, 15:] == PALETTE[8]).all() and (imgs[2] == PALETTE[5]).all()


def test_class_table_matches_palette():
    assert len(CLASS_NAMES) == 19 and PALETTE.shape == (19, 3)
