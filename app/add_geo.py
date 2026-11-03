"""
Add each demo patch's georeference to the demo data so the app can place it on a real map.
Needs pyproj and PASTIS metadata.geojson (in PASTIS.zip, zenodo.org/records/5012942); run once after precompute (no arguments runs the self-check):
    python -m app.add_geo --metadata <PASTIS>/metadata.geojson --data app/data/demo_patches.npz
"""
import argparse
import json
import numpy as np
from pyproj import Transformer

PATCH = 128     # PASTIS patch side in pixels; the demo shows its top-left 120 x 120 (5 x 24 windows)


def patch_corners(ring):
    """top-left, top-right, bottom-left corner of a slightly rotated square ring (N, 2) of x (east), y (north)"""
    s, d = ring[:, 0] + ring[:, 1], ring[:, 0] - ring[:, 1]
    return ring[d.argmin()], ring[s.argmax()], ring[s.argmin()]


def affine(tl, tr, bl, size=PATCH):
    """(2, 3) affine [lon, lat] = geo @ [col, row, 1] from the lon/lat of three corners of a size x size patch"""
    tl, tr, bl = map(np.asarray, (tl, tr, bl))
    return np.stack([(tr - tl) / size, (bl - tl) / size, tl], 1)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--metadata', required=True, help='PASTIS metadata.geojson')
    parser.add_argument('--data', default='app/data/demo_patches.npz')
    args = parser.parse_args(argv)

    meta = json.load(open(args.metadata))
    crs = meta['crs']['properties']['name'].split(':')[-1]          # urn:ogc:def:crs:EPSG::2154
    to_wgs84 = Transformer.from_crs('EPSG:' + crs, 'EPSG:4326', always_xy=True)
    rings = {f['properties']['ID_PATCH']: np.array(f['geometry']['coordinates'][0][0]) for f in meta['features']}

    with np.load(args.data) as d:
        demo = {k: d[k] for k in d.files}
    demo['geo'] = np.stack([affine(*(to_wgs84.transform(*c) for c in patch_corners(rings[int(pid)])))
                            for pid in demo['patch_ids']])
    np.savez_compressed(args.data, **demo)
    print('added geo for %d patches to %s' % (len(demo['geo']), args.data))


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        main()
    else:   # self-check: python -m app.add_geo
        ring = np.array([[0, 10], [10, 11], [11, 1], [1, 0], [0, 10]], float)      # square turned a few degrees, closed
        tl, tr, bl = patch_corners(ring)
        assert tl.tolist() == [0, 10] and tr.tolist() == [10, 11] and bl.tolist() == [1, 0]
        geo = affine([2.0, 48.0], [2.0 + 128e-4, 48.0], [2.0, 48.0 - 128e-4])
        assert np.allclose(geo @ [128, 128, 1], [2 + 128e-4, 48 - 128e-4])
        print("add_geo self-check passed")
