"""
Crop recognition demo (Streamlit). Shows TSViT_DINOv3 predictions precomputed by app/precompute.py.
    streamlit run app/crop_recognition_app.py
"""
import os
import numpy as np
import streamlit as st
from demo_core import PALETTE, PIXEL_HA, area_table, clip_roi, overlay

DATA = os.environ.get('DEMO_DATA', os.path.join(os.path.dirname(__file__), 'data', 'demo_patches.npz'))
SIZE = 120          # stitched patch side in pixels (10 m each)
ZOOM = 5            # nearest-neighbour upscaling for display


@st.cache_data
def load(path):
    with np.load(path) as d:
        return {k: d[k] for k in d.files}


def show(img):
    return np.repeat(np.repeat(img, ZOOM, axis=0), ZOOM, axis=1)


def result_table(rows, total, background):
    td = '<td style="padding:3px 6px;border:none%s">%s</td>'
    swatch = '<span style="display:inline-block;width:14px;height:14px;border:1px solid #333;background:rgb(%d,%d,%d)"></span>'
    cells = ''.join('<tr>%s%s%s</tr>' % (td % ('', swatch % tuple(PALETTE[c])), td % ('', name),
                                        td % (';text-align:right', '%.2f ha' % ha)) for c, name, ha in rows)
    cells += '<tr>%s%s%s</tr>' % (td % ('', ''), td % ('', '<i>Background / other</i>'),
                                  td % (';text-align:right', '<i>%.2f ha</i>' % background))
    cells += '<tr>%s%s%s</tr>' % (td % (';border-top:1px solid #ccc', ''), td % (';border-top:1px solid #ccc', '<b>Selected area</b>'),
                                  td % (';border-top:1px solid #ccc;text-align:right', '<b>%.2f ha</b>' % total))
    return '<table style="width:100%%;border-collapse:collapse;border:none;font-size:0.9rem">%s</table>' % cells


st.set_page_config(page_title='Crop recognition', layout='wide')
st.markdown('<style>.block-container{padding-top:2.5rem;padding-bottom:0}'
            'h1{font-size:1.9rem !important;padding:0 0 0.5rem 0 !important}</style>', unsafe_allow_html=True)
if not os.path.isfile(DATA):
    st.error('Demo data not found: %s. Create it with app/precompute.py.' % DATA)
    st.stop()
demo = load(DATA)

# three columns (controls | image | result) so the analysis is visible without scrolling
left, middle, right = st.columns([1, 1.45, 0.95], gap='medium')
with left:
    st.title('Crop recognition')
    names = ['Patch %d' % p for p in demo['patch_ids']]
    i = names.index(st.selectbox('Selected area (satellite image, 1.2 km × 1.2 km)', names))
    if st.session_state.get('analyzed') != i:
        st.session_state['analyzed'] = None          # a new image needs a new analysis

    x0, x1 = st.slider('Area of interest: columns (10 m pixels)', 0, SIZE, (30, 90), key='xr')
    y0, y1 = st.slider('Area of interest: rows (10 m pixels)', 0, SIZE, (30, 90), key='yr')
    roi = (x0, y0, x1 - x0, y1 - y0)
    st.caption('Time period 2018-2019: Sep 2018 - Nov 2019, %d Sentinel-2 acquisitions' % demo['n_dates'][i])

    if st.button('Analyze selected area', type='primary', width='stretch'):
        st.session_state['analyzed'] = i
    analyzed = st.session_state.get('analyzed') == i
    compare = st.checkbox('Compare with ground truth', disabled=not analyzed)

with middle:
    rgb = demo['rgb'][i]
    if analyzed and compare:
        a, b = st.columns(2)
        a.image(show(overlay(rgb, demo['pred'][i], roi)), caption='Prediction (TSViT_DINOv3)', width='stretch')
        b.image(show(overlay(rgb, demo['label'][i], roi)), caption='Ground truth (PASTIS)', width='stretch')
    else:
        img = overlay(rgb, demo['pred'][i], roi, alpha=0.75 if analyzed else 0.0)
        x0, y0, x1, y1 = clip_roi(roi, rgb.shape)
        st.image(show(img), caption='Area of interest: %.2f ha' % ((x1 - x0) * (y1 - y0) * PIXEL_HA), width='stretch')

with right:
    st.subheader('Result')
    if analyzed:
        st.markdown(result_table(*area_table(demo['pred'][i], roi)), unsafe_allow_html=True)
    else:
        st.caption('Choose an area and press "Analyze selected area".')
    st.caption('Predictions computed in advance with TSViT_DINOv3 (frozen DINOv3 ViT-S/16 encoder) on PASTIS '
               'test patches; 1 pixel = 10 m × 10 m.')
    st.caption('Data: [PASTIS](https://zenodo.org/records/5012942) (Sainte Fare Garnot & Landrieu, 2021), '
               '[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Changed: 24×24 windows stitched into '
               '120×120 patches, RGB composite of the clearest date, void labels shown as background.')
