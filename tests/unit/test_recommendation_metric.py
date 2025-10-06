import numpy as np
from esnf_mat_analyzer.processing.background_correction.recommendation import score_correction, recommend_method


def test_score_correction_basic():
    # Create an original image with range and a corrected image that's slightly modified
    orig = np.linspace(10, 240, 256*256, dtype=np.uint8).reshape((256, 256))
    # Simulate a correction that preserves mean but reduces dynamic range
    corr = np.clip((orig.astype(float) - 10) * 0.9 + 10, 0, 255).astype(np.uint8)

    score = score_correction(orig, corr)
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


def test_recommend_method_prefers_methods_on_synthetic():
    # Synthetic image: gradient with a bright saturated spot
    img = np.tile(np.linspace(0, 255, 256, dtype=np.uint8), (256, 1))

    # Define simple methods: identity, slight blur, heavy blur
    def identity(i):
        return i

    def slight(i):
        from scipy.ndimage import gaussian_filter
        return gaussian_filter(i, sigma=1)

    def heavy(i):
        from scipy.ndimage import gaussian_filter
        return gaussian_filter(i, sigma=5)

    methods = {'identity': identity, 'slight': slight, 'heavy': heavy}

    best, scores = recommend_method(img, methods, downsample=2)
    # Ensure a best method was returned and scores dict populated
    assert best in methods
    assert set(scores.keys()) == set(methods.keys())
