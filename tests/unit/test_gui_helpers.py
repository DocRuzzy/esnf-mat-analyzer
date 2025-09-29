import numpy as np
from PIL import Image
import pytest

from esnf_mat_analyzer.gui.main_window import MainWindow


def test_to_numpy_rgb_from_rgb_pil():
    img = Image.new('RGB', (10, 10), color=(10, 20, 30))
    mw = MainWindow.__new__(MainWindow)
    arr = mw._to_numpy_rgb(img)
    assert arr.shape == (10, 10, 3)
    assert arr.dtype == np.uint8
    assert (arr[0, 0] == np.array([10, 20, 30])).all()


def test_to_numpy_rgb_from_rgba_pil():
    img = Image.new('RGBA', (8, 6), color=(1, 2, 3, 255))
    mw = MainWindow.__new__(MainWindow)
    arr = mw._to_numpy_rgb(img)
    assert arr.shape == (6, 8, 3)
    assert arr.dtype == np.uint8
    assert (arr[0, 0] == np.array([1, 2, 3])).all()


def test_to_numpy_gray_and_empty():
    img = Image.new('L', (5, 5), color=128)
    mw = MainWindow.__new__(MainWindow)
    gray = mw._to_numpy_gray(img)
    assert gray.shape == (5, 5)
    assert gray.dtype == np.uint8

    with pytest.raises(ValueError):
        mw._to_numpy_rgb(None)


def test_roi_shape_metrics_rectangle_and_thin():
    mw = MainWindow.__new__(MainWindow)
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[10:60, 20:80] = 255
    metrics = mw._roi_shape_metrics(mask)
    assert metrics['area'] > 0
    assert 0.0 <= metrics['solidity'] <= 1.0
    assert 0.0 <= metrics['eccentricity'] <= 1.0

    # Thin rectangle
    thin = np.zeros((100, 100), dtype=np.uint8)
    thin[10:15, 20:95] = 255
    m2 = mw._roi_shape_metrics(thin)
    assert m2['area'] > 0
    assert m2['eccentricity'] > metrics['eccentricity']
