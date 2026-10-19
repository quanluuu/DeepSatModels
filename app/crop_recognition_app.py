"""
Crop recognition demo (Streamlit). Shows TSViT_DINOv3 predictions precomputed by app/precompute.py.
    streamlit run app/crop_recognition_app.py
"""
import os
import numpy as np
import streamlit as st
import folium
from folium.plugins import Draw
from streamlit_folium import st_folium
from demo_core import PALETTE, area_table, clip_roi, overlay, roi_from_geojson

DATA = os.environ.get('DEMO_DATA', os.path.join(os.path.dirname(__file__), 'data', 'demo_patches.npz'))
SIZE = 120                      # stitched patch side in pixels (10 m each)
DEFAULT_ROI = (30, 30, 60, 60)  # x, y, w, h
MAP_HEIGHT = 560


@st.cache_data
def load(path):
    with np.load(path) as d:
        return {k: d[k] for k in d.files}


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


def area_map(img, roi):
    """Leaflet map in pixel coordinates (CRS.Simple: lng = column, lat = SIZE - row) with a rectangle tool"""
    m = folium.Map(crs='Simple', tiles=None, zoom_control=True, attribution_control=False,
                   min_zoom=1, max_zoom=6, max_bounds=True,      # keep panning around the image, not the lat +-90 world
                   min_lat=-SIZE / 4, max_lat=SIZE * 5 / 4, min_lon=-SIZE / 4, max_lon=SIZE * 5 / 4)
    folium.raster_layers.ImageOverlay(img, bounds=[[0, 0], [SIZE, SIZE]], pixelated=True).add_to(m)
    x0, y0, x1, y1 = clip_roi(roi, img.shape)
    folium.Rectangle([[SIZE - y1, x0], [SIZE - y0, x1]], color='white', weight=2, fill=False,
                     tooltip=folium.Tooltip('Selected area', permanent=True, direction='center')).add_to(m)
    Draw(draw_options={'rectangle': {'shapeOptions': {'color': '#ffffff', 'weight': 2, 'fillOpacity': 0.1}},
                       'polyline': False, 'polygon': False, 'circle': False, 'marker': False, 'circlemarker': False},
         edit_options={'edit': False, 'remove': False}).add_to(m)
    m.fit_bounds([[0, 0], [SIZE, SIZE]])
    return m


st.set_page_config(page_title='Crop recognition', layout='wide')
st.markdown('<style>.block-container{padding-top:2.5rem;padding-bottom:0}'
            'h1{font-size:1.9rem !important;padding:0 0 0.2rem 0 !important}</style>', unsafe_allow_html=True)
if not os.path.isfile(DATA):
    st.error('Demo data not found: %s. Create it with app/precompute.py.' % DATA)
    st.stop()
demo = load(DATA)
names = ['Patch %d' % p for p in demo['patch_ids']]
state = st.session_state
state.setdefault('roi', DEFAULT_ROI)
state.setdefault('map_gen', 0)      # bumped to clear rectangles drawn on the map

left, right = st.columns([1, 2.1], gap='large')
with right:
    hint, whole, region = st.columns([1.6, 1, 1.6], vertical_alignment='bottom')
    hint.caption('Draw a rectangle with the ▭ tool on the map to select an area')
    i = names.index(region.selectbox('Available region (1.2 km × 1.2 km)', names, key='region'))
    if state.get('patch') != i:      # a new image needs a new area and a new analysis
        state.update(patch=i, roi=DEFAULT_ROI, analyzed=False, map_gen=state['map_gen'] + 1)
    if whole.button('Whole region', key='whole', width='stretch'):
        state.update(roi=(0, 0, SIZE, SIZE), map_gen=state['map_gen'] + 1)

with left:
    st.title('Crop recognition')
    st.caption('Identify crops from satellite time series')
    a, s = st.columns(2, vertical_alignment='bottom')
    if a.button('Reselect area', key='reselect', width='stretch'):
        state.update(roi=DEFAULT_ROI, map_gen=state['map_gen'] + 1)
    s.selectbox('Season', ['2018-2019'], disabled=True)
    st.caption('Sentinel-2 · Sep 2018 - Nov 2019 · 10 m imagery')
    if st.button('Analyze selected area', key='analyze', type='primary', width='stretch'):
        state['analyzed'] = True
    show_gt = st.toggle('Show ground truth (PASTIS)', key='gt', disabled=not state.get('analyzed'))

with right:
    roi = state['roi']
    labels = demo['label'][i] if show_gt else demo['pred'][i]
    img = overlay(demo['rgb'][i], labels, roi, alpha=0.75 if state.get('analyzed') else 0.0)
    out = st_folium(area_map(img, roi), key='map-%d-%d' % (i, state['map_gen']), height=MAP_HEIGHT,
                    use_container_width=True, returned_objects=['last_active_drawing'])
    drawn = roi_from_geojson((out or {}).get('last_active_drawing'))
    if drawn and drawn != roi:
        state['roi'] = drawn
        st.rerun()

with left:
    rows, total, background = area_table(demo['pred'][i], roi)
    with st.container(border=True):
        if state.get('analyzed'):
            st.markdown('**Result%s**' % (' (prediction)' if show_gt else ''))
            st.markdown(result_table(rows, total, background), unsafe_allow_html=True)
        else:
            st.markdown('**Ready to analyze**')
            st.caption('%.2f ha · %d observations per location. TSViT_DINOv3 labels every 10 m pixel with one of the '
                       '19 PASTIS classes, including background.' % (total, demo['n_dates'][i]))
    st.caption('Predictions computed in advance with TSViT_DINOv3 (frozen DINOv3 ViT-S/16 encoder) on PASTIS test '
               'patches. Data: [PASTIS](https://zenodo.org/records/5012942) (Sainte Fare Garnot & Landrieu, 2021), '
               '[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); 24×24 windows stitched into 120×120 '
               'patches, RGB composite of the clearest date, void labels shown as background.')
