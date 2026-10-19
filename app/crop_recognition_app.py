"""
Crop recognition demo (Streamlit). Shows TSViT_DINOv3 predictions precomputed by app/precompute.py.
    streamlit run app/crop_recognition_app.py
"""
import os
import numpy as np
import streamlit as st
from demo_core import CLASS_NAMES, PALETTE, PIXEL_HA, area_table, clip_roi, overlay

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
    cells = ''.join(
        '<tr><td><span style="display:inline-block;width:18px;height:18px;border:1px solid #333;'
        'background:rgb(%d,%d,%d)"></span></td><td>%s</td><td style="text-align:right">%.2f ha</td></tr>'
        % (*PALETTE[c], name, ha) for c, name, ha in rows)
    cells += ('<tr><td></td><td><i>Background / other</i></td><td style="text-align:right"><i>%.2f ha</i></td></tr>'
              '<tr><td></td><td><b>Selected area</b></td><td style="text-align:right"><b>%.2f ha</b></td></tr>'
              % (background, total))
    return '<table style="width:100%%;border:none">%s</table>' % cells


st.set_page_config(page_title='Crop recognition', layout='wide')
if not os.path.isfile(DATA):
    st.error('Demo data not found: %s. Create it with app/precompute.py.' % DATA)
    st.stop()
demo = load(DATA)

left, right = st.columns([1, 2.4], gap='large')
with left:
    st.title('Crop recognition')
    st.subheader('Selected area')
    names = ['Patch %d' % p for p in demo['patch_ids']]
    i = names.index(st.selectbox('Satellite image (1.2 km × 1.2 km)', names))
    if st.session_state.get('analyzed') != i:
        st.session_state['analyzed'] = None          # a new image needs a new analysis

    st.caption('Area of interest, in 10 m pixels')
    c1, c2 = st.columns(2)
    x = c1.slider('x', 0, SIZE - 1, 30, key='x')
    y = c2.slider('y', 0, SIZE - 1, 30, key='y')
    w = c1.slider('width', 1, SIZE, 60, key='w')
    h = c2.slider('height', 1, SIZE, 60, key='h')
    roi = (x, y, w, h)

    st.subheader('Time period')
    st.selectbox('Season', ['2018-2019'], disabled=True, label_visibility='collapsed')
    st.caption('Time sequence: Sep 2018 - Nov 2019 (%d Sentinel-2 acquisitions)' % demo['n_dates'][i])

    if st.button('Analyze selected area', type='primary', width='stretch'):
        st.session_state['analyzed'] = i
    analyzed = st.session_state.get('analyzed') == i
    compare = st.checkbox('Compare with ground truth', disabled=not analyzed)

    st.divider()
    st.subheader('Result')
    if analyzed:
        st.markdown(result_table(*area_table(demo['pred'][i], roi)), unsafe_allow_html=True)
    else:
        st.caption('Choose an area and press "Analyze selected area".')

with right:
    rgb = demo['rgb'][i]
    if analyzed and compare:
        a, b = st.columns(2)
        a.image(show(overlay(rgb, demo['pred'][i], roi)), caption='Prediction (TSViT_DINOv3)', width='stretch')
        b.image(show(overlay(rgb, demo['label'][i], roi)), caption='Ground truth (PASTIS)', width='stretch')
    else:
        img = overlay(rgb, demo['pred'][i], roi, alpha=0.75 if analyzed else 0.0)
        x0, y0, x1, y1 = clip_roi(roi, rgb.shape)
        st.image(show(img), caption='Area of interest: %.2f ha' % ((x1 - x0) * (y1 - y0) * PIXEL_HA), width='stretch')
    st.caption('Predictions were computed in advance with TSViT_DINOv3 (frozen DINOv3 ViT-S/16 encoder) on '
               'PASTIS test patches; 1 pixel = 10 m × 10 m. Classes: %s.' % ', '.join(CLASS_NAMES[1:]))
