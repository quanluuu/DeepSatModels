"""system tests for the Streamlit demo (streamlit.testing AppTest; the st_folium map itself is not driven)"""
import os
import numpy as np
from streamlit.testing.v1 import AppTest

APP = os.path.join(os.path.dirname(__file__), '..', 'app', 'crop_recognition_app.py')
TIMEOUT = 30


def run_app(monkeypatch, data=None):
    if data is not None:
        monkeypatch.setenv('DEMO_DATA', str(data))
    return AppTest.from_file(APP, default_timeout=TIMEOUT).run()


def test_renders_with_real_demo_data(monkeypatch):
    at = run_app(monkeypatch)
    assert not at.exception
    assert not at.error
    assert at.title[0].value == 'Crop recognition'
    assert len(at.selectbox(key='region').options) == 25


def test_analyze_shows_result_table(monkeypatch):
    at = run_app(monkeypatch)
    at.button(key='analyze').click().run()
    assert not at.exception
    assert any('Result (prediction)' in m.value for m in at.markdown)
    assert len(at.get('image')) == 3     # RGB, prediction, ground truth


def test_missing_data_file_shows_error(monkeypatch, tmp_path):
    at = run_app(monkeypatch, tmp_path / 'missing.npz')
    assert not at.exception
    assert 'Demo data not found' in at.error[0].value


def test_data_without_geo_shows_error(monkeypatch, tmp_path):
    path = tmp_path / 'no_geo.npz'
    np.savez(path, patch_ids=np.arange(1), pred=np.zeros((1, 120, 120), np.uint8))
    at = run_app(monkeypatch, path)
    assert not at.exception
    assert 'no patch locations' in at.error[0].value
