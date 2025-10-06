import pytest
from esnf_mat_analyzer.gui.main_window import MainWindow


def test_compute_pixels_per_mm_valid():
    mw = MainWindow.__new__(MainWindow)
    assert mw.compute_pixels_per_mm(500, 10) == 50.0


def test_compute_pixels_per_mm_zero_mm():
    mw = MainWindow.__new__(MainWindow)
    with pytest.raises(ValueError):
        mw.compute_pixels_per_mm(100, 0)


def test_compute_pixels_per_mm_negative_mm():
    mw = MainWindow.__new__(MainWindow)
    with pytest.raises(ValueError):
        mw.compute_pixels_per_mm(100, -5)


def test_compute_pixels_per_mm_none_mm():
    mw = MainWindow.__new__(MainWindow)
    with pytest.raises(ValueError):
        mw.compute_pixels_per_mm(100, None)
