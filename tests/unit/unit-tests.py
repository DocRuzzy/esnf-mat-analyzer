"""
Unit tests for nanofiber thickness uniformity analysis.

This module contains tests for the core components of the nanofiber analyzer.
"""

import unittest
import numpy as np
from pathlib import Path
import tempfile
import shutil
import os

# Import components to test
from nanofiber_analyzer.config.config_manager import (
    Config, ProcessingConfig, CircleDetectionConfig, 
    ThicknessConfig, UniformityConfig, VisualizationConfig
)
from nanofiber_analyzer.processing.image_processor import ImageProcessor
from nanofiber_analyzer.processing.circle_detector import CircleDetector
from nanofiber_analyzer.processing.thickness_estimator import ThicknessEstimator
from nanofiber_analyzer.analysis.radial_uniformity import RadialUniformityIndex
from nanofiber_analyzer.analysis.gini_coefficient import GiniCoefficient
from nanofiber_analyzer.analysis.thickness_ratio import ThicknessRangeRatio
from nanofiber_analyzer.visualization.visualizer import Visualizer
from nanofiber_analyzer.utils.data_exporter import DataExporter


class TestImageProcessor(unittest.TestCase):
    """Tests for the ImageProcessor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = ProcessingConfig()
        self.processor = ImageProcessor(self.config)
        
        # Create a temporary test image
        self.temp_dir = tempfile.mkdtemp()
        self.test_image_path = Path(self.temp_dir) / "test_image.png"
        
        # Create a simple test image (black background with white circle)
        image = np.zeros((200, 200, 3), dtype=np.uint8)
        # Draw white circle
        center = (100, 100)
        radius = 50
        y, x = np.ogrid[:200, :200]
        mask = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= radius ** 2
        image[mask] = 255
        
        # Save the test image
        import cv2
        cv2.imwrite(str(self.test_image_path), image)
    
    def tearDown(self):
        """Tear down test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_load_image(self):
        """Test loading an image from file."""
        image = self.processor.load_image(self.test_image_path)
        
        # Check dimensions and type
        self.assertEqual(image.shape, (200, 200, 3))
        self.assertEqual(image.dtype, np.uint8)
        
        # Check RGB conversion (OpenCV loads as BGR)
        # Center should be white (255, 255, 255) in RGB
        center_pixel = image[100, 100]
        self.assertTrue(np.array_equal(center_pixel, [255, 255, 255]))
    
    def test_preprocess(self):
        """Test image preprocessing."""
        # Load the test image
        image = self.processor.load_image(self.test_image_path)
        
        # Preprocess the image
        processed = self.processor.preprocess(image)
        
        # Check that the output is grayscale
        self.assertEqual(len(processed.shape), 2)
        self.assertEqual(processed.dtype, np.uint8)
        
        # Check that the circle is preserved in the grayscale image
        # Center should be white (255) in grayscale
        self.assertEqual(processed[100, 100], 255)
        # Corner should be black (0) in grayscale
        self.assertEqual(processed[0, 0], 0)


class TestCircleDetector(unittest.TestCase):
    """Tests for the CircleDetector class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = CircleDetectionConfig()
        self.detector = CircleDetector(self.config)
        
        # Create a test image with a circle
        self.image = np.zeros((200, 200), dtype=np.uint8)
        # Draw white circle
        self.center = (100, 100)
        self.radius = 50
        y, x = np.ogrid[:200, :200]
        mask = (x - self.center[0]) ** 2 + (y - self.center[1]) ** 2 <= self.radius ** 2
        self.image[mask] = 255
    
    def test_detect_hough(self):
        """Test circle detection using Hough Transform."""
        # Set detection method to Hough
        self.detector.config.detection_method = "hough"
        
        # Detect circle
        center, radius = self.detector.detect(self.image)
        
        # Check that the detected center and radius are close to the actual values
        self.assertAlmostEqual(center[0], self.center[0], delta=5)
        self.assertAlmostEqual(center[1], self.center[1], delta=5)
        self.assertAlmostEqual(radius, self.radius, delta=5)
    
    def test_detect_contour(self):
        """Test circle detection using contour-based method."""
        # Set detection method to contour
        self.detector.config.detection_method = "contour"
        
        # Detect circle
        center, radius = self.detector.detect(self.image)
        
        # Check that the detected center and radius are close to the actual values
        self.assertAlmostEqual(center[0], self.center[0], delta=5)
        self.assertAlmostEqual(center[1], self.center[1], delta=5)
        self.assertAlmostEqual(radius, self.radius, delta=5)
    
    def test_create_mask(self):
        """Test creating a circular mask."""
        # Create mask
        mask = self.detector.create_mask(self.image.shape, self.center, self.radius)
        
        # Check mask dimensions and type
        self.assertEqual(mask.shape, self.image.shape)
        self.assertEqual(mask.dtype, np.uint8)
        
        # Check that the mask contains the circle
        # Center should be 1
        self.assertEqual(mask[self.center[1], self.center[0]], 1)
        # Corner should be 0
        self.assertEqual(mask[0, 0], 0)


class TestThicknessEstimator(unittest.TestCase):
    """Tests for the ThicknessEstimator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = ThicknessConfig()
        self.estimator = ThicknessEstimator(self.config)
        
        # Create a test image with a gradient circle
        self.image = np.zeros((200, 200), dtype=np.uint8)
        self.center = (100, 100)
        self.radius = 50
        
        # Create a radial gradient in the circle
        y, x = np.ogrid[:200, :200]
        dist = np.sqrt((x - self.center[0]) ** 2 + (y - self.center[1]) ** 2)
        # Normalize distances to [0, 1] range within the circle
        normalized_dist = np.clip(dist / self.radius, 0, 1)
        # Invert and scale to [0, 255] range
        gradient = (1 - normalized_dist) * 255
        # Apply the circle mask
        mask = dist <= self.radius
        self.image[mask] = gradient[mask].astype(np.uint8)
        
        # Create a binary mask for the circle
        self.mask = np.zeros_like(self.image, dtype=np.uint8)
        self.mask[mask] = 1
    
    def test_estimate_linear(self):
        """Test thickness estimation using linear model."""
        # Set model type to linear
        self.estimator.config.model_type = "linear"
        self.estimator.config.a = 1.0
        self.estimator.config.b = 0.0
        
        # Estimate thickness
        thickness_map = self.estimator.estimate(self.image, self.mask)
        
        # Check thickness map dimensions
        self.assertEqual(thickness_map.shape, self.image.shape)
        
        # Check that thickness at center is higher than at the edge
        center_thickness = thickness_map[self.center[1], self.center[0]]
        edge_point = (self.center[0] + int(self.radius * 0.9), self.center[1])
        edge_thickness = thickness_map[edge_point[1], edge_point[0]]
        self.assertGreater(center_thickness, edge_thickness)
    
    def test_get_saturation_mask(self):
        """Test creating a saturation mask."""
        # Set saturation threshold
        self.estimator.config.saturation_threshold = 200
        
        # Get saturation mask
        saturation_mask = self.estimator.get_saturation_mask(self.image, self.mask)
        
        # Check mask dimensions
        self.assertEqual(saturation_mask.shape, self.image.shape)
        
        # Check that center is saturated (value > threshold)
        self.assertEqual(saturation_mask[self.center[1], self.center[0]], 1)
        
        # Check that boundary is not saturated
        edge_point = (self.center[0] + int(self.radius * 0.9), self.center[1])
        self.assertEqual(saturation_mask[edge_point[1], edge_point[0]], 0)


class TestUniformityMetrics(unittest.TestCase):
    """Tests for uniformity metrics classes."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create config for RadialUniformityIndex
        self.uniformity_config = UniformityConfig(num_radial_lines=12)
        
        # Create uniformity metrics
        self.radial_uniformity = RadialUniformityIndex(self.uniformity_config)
        self.gini_coefficient = GiniCoefficient()
        self.thickness_ratio = ThicknessRangeRatio()
        
        # Create a test thickness map with uniform thickness
        self.uniform_map = np.ones((200, 200), dtype=float) * 100
        
        # Create a test thickness map with non-uniform thickness
        self.non_uniform_map = np.ones((200, 200), dtype=float) * 50
        # Add a thicker region
        self.non_uniform_map[50:150, 50:150] = 150
        
        # Create a binary mask for the analysis region
        self.mask = np.zeros((200, 200), dtype=np.uint8)
        self.mask[25:175, 25:175] = 1
        
        # Center of the analysis region
        self.center = (100, 100)
    
    def test_radial_uniformity_index(self):
        """Test the RadialUniformityIndex class."""
        # Calculate RUI for uniform map
        uniform_rui = self.radial_uniformity.calculate(
            self.uniform_map, self.mask, self.center
        )
        
        # Calculate RUI for non-uniform map
        non_uniform_rui = self.radial_uniformity.calculate(
            self.non_uniform_map, self.mask, self.center
        )
        
        # Check that uniform map has higher RUI (closer to 1)
        self.assertGreater(uniform_rui, non_uniform_rui)
        self.assertGreaterEqual(uniform_rui, 0.0)
        self.assertLessEqual(uniform_rui, 1.0)
    
    def test_gini_coefficient(self):
        """Test the GiniCoefficient class."""
        # Calculate Gini for uniform map
        uniform_gini = self.gini_coefficient.calculate(
            self.uniform_map, self.mask, self.center
        )
        
        # Calculate Gini for non-uniform map
        non_uniform_gini = self.gini_coefficient.calculate(
            self.non_uniform_map, self.mask, self.center
        )
        
        # Check that uniform map has lower Gini (closer to 0)
        self.assertLess(uniform_gini, non_uniform_gini)
        self.assertGreaterEqual(uniform_gini, 0.0)
        self.assertLessEqual(uniform_gini, 1.0)
    
    def test_thickness_range_ratio(self):
        """Test the ThicknessRangeRatio class."""
        # Calculate TRR for uniform map
        uniform_trr = self.thickness_ratio.calculate(
            self.uniform_map, self.mask, self.center
        )
        
        # Calculate TRR for non-uniform map
        non_uniform_trr = self.thickness_ratio.calculate(
            self.non_uniform_map, self.mask, self.center
        )
        
        # Check that uniform map has higher TRR (closer to 1)
        self.assertGreater(uniform_trr, non_uniform_trr)
        self.assertGreaterEqual(uniform_trr, 0.0)
        self.assertLessEqual(uniform_trr, 1.0)
        # For completely uniform map, TRR should be 1.0
        self.assertAlmostEqual(uniform_trr, 1.0)


class TestDataExporter(unittest.TestCase):
    """Tests for the DataExporter class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.exporter = DataExporter()
        
        # Create a simple thickness map for testing
        self.thickness_map = np.ones((10, 10), dtype=float) * 100
        self.thickness_map[3:7, 3:7] = 200
        
        # Create test metrics
        self.metrics = {
            'Radial Uniformity Index': 0.85,
            'Gini Coefficient': 0.25,
            'Thickness Range Ratio': 0.5
        }
        
        # Create temporary directory for exports
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Tear down test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_export_thickness_map(self):
        """Test exporting thickness map to CSV."""
        # Create output path
        output_path = Path(self.temp_dir) / "thickness_map.csv"
        
        # Export thickness map
        self.exporter.export_thickness_map(self.thickness_map, output_path)
        
        # Check that the file was created
        self.assertTrue(output_path.exists())
        
        # Check file content (basic validation)
        with open(output_path, 'r') as f:
            lines = f.readlines()
            
            # Check header
            self.assertTrue(lines[0].startswith('y/x'))
            
            # Check that we have the right number of rows
            self.assertEqual(len(lines), self.thickness_map.shape[0] + 1)
    
    def test_export_metrics(self):
        """Test exporting metrics to JSON."""
        # Create output path
        output_path = Path(self.temp_dir) / "metrics.json"
        
        # Export metrics
        self.exporter.export_metrics(self.metrics, output_path)
        
        # Check that the file was created
        self.assertTrue(output_path.exists())
        
        # Check file content
        import json
        with open(output_path, 'r') as f:
            loaded_metrics = json.load(f)
            
            # Check that the metrics match
            for key, value in self.metrics.items():
                self.assertIn(key, loaded_metrics)
                self.assertEqual(loaded_metrics[key], value)


class TestEndToEnd(unittest.TestCase):
    """End-to-end test of the full analysis pipeline."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for tests
        self.temp_dir = tempfile.mkdtemp()
        
        # Create a test image with a gradient circle
        self.image_path = Path(self.temp_dir) / "test_nanofiber.png"
        
        # Create the image
        image = np.zeros((200, 200, 3), dtype=np.uint8)
        center = (100, 100)
        radius = 50
        
        # Create a radial gradient in the circle
        y, x = np.ogrid[:200, :200]
        dist = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
        # Normalize distances to [0, 1] range within the circle
        normalized_dist = np.clip(dist / radius, 0, 1)
        # Invert and scale to [0, 255] range
        gradient = (1 - normalized_dist) * 255
        # Apply the circle mask
        mask = dist <= radius
        for i in range(3):  # Apply to all RGB channels
            image[..., i][mask] = gradient[mask].astype(np.uint8)
        
        # Save the test image
        import cv2
        cv2.imwrite(str(self.image_path), image)
        
        # Create configuration
        self.config = Config(
            processing=ProcessingConfig(),
            circle_detection=CircleDetectionConfig(),
            thickness=ThicknessConfig(),
            uniformity=UniformityConfig(),
            visualization=VisualizationConfig(),
            output_dir=Path(self.temp_dir) / "output"
        )
        
        # Create analyzer components
        from nanofiber_analyzer.main import setup_dependencies
        self.analyzer = setup_dependencies(self.config)
    
    def tearDown(self):
        """Tear down test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_process_image(self):
        """Test processing a single image through the full pipeline."""
        # Process the test image
        result = self.analyzer.process_image(self.image_path)
        
        # Check that the result contains expected keys
        self.assertIn('image_path', result)
        self.assertIn('center', result)
        self.assertIn('radius', result)
        self.assertIn('metrics', result)
        self.assertIn('results_dir', result)
        
        # Check that all metrics were calculated
        metrics = result['metrics']
        self.assertIn('Radial Uniformity Index', metrics)
        self.assertIn('Gini Coefficient', metrics)
        self.assertIn('Thickness Range Ratio', metrics)
        
        # Check that output files were created
        results_dir = Path(result['results_dir'])
        self.assertTrue((results_dir / "thickness_heatmap.png").exists())
        self.assertTrue((results_dir / "radial_profile.png").exists())
        self.assertTrue((results_dir / "uniformity_metrics.png").exists())
        self.assertTrue((results_dir / "thickness_map.csv").exists())
        self.assertTrue((results_dir / "metrics.json").exists())
        self.assertTrue((results_dir / "summary.txt").exists())


if __name__ == '__main__':
    unittest.main()
