import unittest
import numpy as np

from esnf_mat_analyzer.processing.background_correction.advanced import AdvancedBackgroundProcessor
from esnf_mat_analyzer.processing.advanced_thickness import AdvancedThicknessEstimator
from esnf_mat_analyzer.analysis.frequency_metrics import FrequencyAnalyzer
from esnf_mat_analyzer.analysis.texture_metrics import GLCMAnalyzer, LBPAnalyzer, FractalAnalyzer

class TestAdvancedFeatures(unittest.TestCase):

    def setUp(self):
        self.adv_background_processor = AdvancedBackgroundProcessor()
        self.adv_thickness_estimator = AdvancedThicknessEstimator()
        self.freq_analyzer = FrequencyAnalyzer()
        self.glcm_analyzer = GLCMAnalyzer()
        self.lbp_analyzer = LBPAnalyzer()
        self.fractal_analyzer = FractalAnalyzer()
        self.test_image = np.random.randint(0, 256, (100, 100), dtype=np.uint8)

    def test_basic_correction(self):
        corrected_image = self.adv_background_processor.basic_correction(self.test_image)
        self.assertEqual(corrected_image.shape, self.test_image.shape)
        self.assertEqual(corrected_image.dtype, np.uint8)

    def test_rolling_ball_3d(self):
        corrected_image = self.adv_background_processor.rolling_ball_3d(self.test_image, radius=10)
        self.assertEqual(corrected_image.shape, self.test_image.shape)
        self.assertEqual(corrected_image.dtype, np.uint8)

    def test_restore_method(self):
        corrected_image = self.adv_background_processor.restore_method(self.test_image)
        self.assertEqual(corrected_image.shape, self.test_image.shape)
        self.assertEqual(corrected_image.dtype, np.uint8)

    def test_homomorphic_filter(self):
        corrected_image = self.adv_background_processor.homomorphic_filter(self.test_image)
        self.assertEqual(corrected_image.shape, self.test_image.shape)
        self.assertEqual(corrected_image.dtype, np.uint8)

    def test_beer_lambert_estimation(self):
        transmittance = np.random.rand(100, 100)
        thickness = self.adv_thickness_estimator.beer_lambert_estimation(transmittance, attenuation_coefficient=0.1)
        self.assertEqual(thickness.shape, transmittance.shape)

    def test_calculate_anisotropy_index(self):
        anisotropy_index = self.freq_analyzer.calculate_anisotropy_index(self.test_image)
        self.assertIsInstance(anisotropy_index, float)

    def test_power_spectral_density(self):
        psd = self.freq_analyzer.power_spectral_density(self.test_image)
        self.assertEqual(psd.shape, self.test_image.shape)

    def test_calculate_glcm_features(self):
        features = self.glcm_analyzer.calculate_glcm_features(self.test_image)
        self.assertIsInstance(features, dict)
        self.assertIn('contrast', features)
        self.assertIn('correlation', features)
        self.assertIn('energy', features)
        self.assertIn('homogeneity', features)

    def test_calculate_lbp_uniformity(self):
        uniformity = self.lbp_analyzer.calculate_lbp_uniformity(self.test_image)
        self.assertIsInstance(uniformity, float)

    def test_calculate_fractal_dimension(self):
        fractal_dimension = self.fractal_analyzer.calculate_fractal_dimension(self.test_image)
        self.assertIsInstance(fractal_dimension, float)

if __name__ == '__main__':
    unittest.main()
