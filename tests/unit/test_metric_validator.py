import math
from esnf_mat_analyzer.validation.metric_validator import validate_metrics


def test_validate_metrics_basic():
    metrics = {
        'overall_mat_uniformity': 0.75,
        'anisotropy_index': 0.1,
        'texture_contrast': 2.5,
        'texture_correlation': 0.2,
        'gini': 0.4
    }
    _, issues = validate_metrics(metrics)
    assert issues == {}


def test_validate_metrics_out_of_range_and_nan():
    metrics = {
        'overall_mat_uniformity': 1.5,  # out of 0..1
        'texture_correlation': float('nan'),
        'texture_contrast': -1.0,
        'some_other': float('inf')
    }
    _, issues = validate_metrics(metrics)
    # Expect issues for each invalid metric
    assert 'overall_mat_uniformity' in issues
    assert 'texture_correlation' in issues
    assert 'texture_contrast' in issues
    assert 'some_other' in issues
