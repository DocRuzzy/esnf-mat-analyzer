from types import SimpleNamespace
import math

from esnf_mat_analyzer.gui.main_window import MainWindow


def make_result(metrics):
    # Minimal object with the attributes used by _format_mat_results
    return SimpleNamespace(metrics=metrics, image_path="/tmp/img.png", processing_time=0.12, spatial_scale_pixels_per_mm=5.0)


def test_format_mat_results_with_valid_metrics():
    mw = MainWindow.__new__(MainWindow)
    metrics = {
        'overall_mat_uniformity': 0.72,
        'anisotropy_index': 0.05,
        'texture_homogeneity': 0.8,
        'psd_uniformity': 0.4
    }
    res = make_result(metrics)
    text = mw._format_mat_results(res)
    assert 'Overall Mat Uniformity Score' in text
    assert 'Anisotropy Index' in text
    assert 'Texture-based Uniformity' in text


def test_format_mat_results_with_invalid_metrics():
    mw = MainWindow.__new__(MainWindow)
    metrics = {
        'overall_mat_uniformity': float('nan'),
        'texture_contrast': -5.0,
        'texture_correlation': 2.0
    }
    res = make_result(metrics)
    text = mw._format_mat_results(res)
    assert 'METRIC ISSUES DETECTED' in text
    # Ensure issues mention the invalid keys
    assert 'texture_contrast' in text
    assert 'texture_correlation' in text
