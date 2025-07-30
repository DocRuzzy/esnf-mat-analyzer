"""
Tests for the ImageProcessor class.

This module contains unit tests for the ImageProcessor class to ensure
it correctly loads and preprocesses images for nanofiber analysis.
"""

import pytest
import numpy as np
import tempfile
import os
from pathlib import Path
import cv2

from esnf_mat_analyzer.core.data_types import (
    ProcessingConfig,
    GrayscaleConversionMethod,
)
from esnf_mat_analyzer.processing.image_processor import ImageProcessor


class TestImageProcessor:
    """Test suite for the ImageProcessor class."""

    @pytest.fixture
    def test_image_path(self):
        """Create a temporary test image and return its path."""
        # Create a temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a simple test image (black with white circle)
            image = np.zeros((200, 200, 3), dtype=np.uint8)
            # Draw white circle in the center
            cv2.circle(image, (100, 100), 50, (255, 255, 255), -1)
            # Save the image
            image_path = Path(temp_dir) / "test_image.png"
            cv2.imwrite(str(image_path), image)

            yield image_path

    @pytest.fixture
    def default_config(self):
        """Return a default processing configuration."""
        return ProcessingConfig()

    def test_load_image(self, test_image_path, default_config):
        """Test that images can be loaded correctly."""
        processor = ImageProcessor(default_config)
        image = processor.load_image(test_image_path)

        # Check that the image was loaded with the right dimensions and type
        assert isinstance(image, np.ndarray)
        assert image.shape == (200, 200, 3)
        assert image.dtype == np.uint8

        # Check that the image was converted to RGB (white circle should be white in RGB)
        assert np.array_equal(image[100, 100], [255, 255, 255])

    def test_load_image_nonexistent(self, default_config):
        """Test that loading a non-existent image raises an error."""
        processor = ImageProcessor(default_config)
        with pytest.raises(FileNotFoundError):
            processor.load_image(Path("nonexistent_image.png"))

    def test_preprocess_with_default_config(self, test_image_path, default_config):
        """Test image preprocessing with default configuration."""
        default_config.leveling.enabled = False
        processor = ImageProcessor(default_config)
        image = processor.load_image(test_image_path)
        preprocessed = processor.preprocess(image)

        # Check that the output is grayscale
        assert len(preprocessed.shape) == 2
        assert preprocessed.dtype == np.uint8

        # Center should be white (255)
        assert preprocessed[100, 100] > 240
        # Corner should be black (0)
        assert preprocessed[0, 0] < 10

    def test_grayscale_conversion_methods(self, test_image_path):
        """Test different grayscale conversion methods."""
        # Create a color image with different colors
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        # Red square
        image[0:50, 0:50] = [255, 0, 0]
        # Green square
        image[0:50, 50:100] = [0, 255, 0]
        # Blue square
        image[50:100, 0:50] = [0, 0, 255]
        # White square
        image[50:100, 50:100] = [255, 255, 255]

        # Save the image
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "color_test.png"
            cv2.imwrite(str(image_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

            # Test weighted method
            weighted_config = ProcessingConfig(
                grayscale_conversion=GrayscaleConversionMethod.WEIGHTED
            )
            weighted_config.leveling.enabled = False
            weighted_processor = ImageProcessor(weighted_config)
            weighted_image = weighted_processor.load_image(image_path)
            weighted_gray = weighted_processor.preprocess(weighted_image)

            # Test average method
            average_config = ProcessingConfig(
                grayscale_conversion=GrayscaleConversionMethod.AVERAGE
            )
            average_config.leveling.enabled = False
            average_processor = ImageProcessor(average_config)
            average_image = average_processor.load_image(image_path)
            average_gray = average_processor.preprocess(average_image)

            # Test luminance method
            luminance_config = ProcessingConfig(
                grayscale_conversion=GrayscaleConversionMethod.LUMINANCE
            )
            luminance_config.leveling.enabled = False
            luminance_processor = ImageProcessor(luminance_config)
            luminance_image = luminance_processor.load_image(image_path)
            luminance_gray = luminance_processor.preprocess(luminance_image)

            # Verify that the methods produce different results
            # Red square should be darker in luminance method than in average method
            assert luminance_gray[25, 25] < average_gray[25, 25]
            # Green square should be brighter in luminance method than red square
            assert luminance_gray[25, 75] > luminance_gray[25, 25]
            # White square should be white in all methods
            assert weighted_gray[75, 75] > 240
            assert average_gray[75, 75] > 240
            assert luminance_gray[75, 75] > 240

    def test_blur_and_contrast(self, test_image_path):
        """Test blur and contrast adjustments."""
        # Load the original image
        config = ProcessingConfig()
        config.leveling.enabled = False
        default_processor = ImageProcessor(config)
        original = default_processor.load_image(test_image_path)

        # Create a configuration with strong blur
        blur_config = ProcessingConfig(blur_kernel_size=15)
        blur_config.leveling.enabled = False
        blur_processor = ImageProcessor(blur_config)
        blurred = blur_processor.preprocess(original)

        # Create a configuration with increased contrast
        contrast_config = ProcessingConfig(contrast_alpha=2.0)
        contrast_config.leveling.enabled = False
        contrast_processor = ImageProcessor(contrast_config)
        contrasted = contrast_processor.preprocess(original)

        # Create a configuration with no blur
        no_blur_config = ProcessingConfig(blur_kernel_size=0)
        no_blur_config.leveling.enabled = False
        no_blur_processor = ImageProcessor(no_blur_config)
        no_blur = no_blur_processor.preprocess(original)

        # Verify that blur reduces edge sharpness
        # Calculate edge strength using Sobel operator
        blurred_edges = cv2.Sobel(blurred, cv2.CV_64F, 1, 1, ksize=3)
        no_blur_edges = cv2.Sobel(no_blur, cv2.CV_64F, 1, 1, ksize=3)

        # Blurred image should have weaker edges
        assert np.max(np.abs(blurred_edges)) < np.max(np.abs(no_blur_edges))

        # Verify that increased contrast enhances the difference between black and white
        default = default_processor.preprocess(original)

        # Standard deviation should be higher with increased contrast
        assert np.std(contrasted) > np.std(default)

    def test_analyze_image_properties(self, test_image_path, default_config):
        """Test the image property analysis function."""
        processor = ImageProcessor(default_config)
        image = processor.load_image(test_image_path)

        properties = processor.analyze_image_properties(image)

        # Check that all expected properties are present
        expected_keys = [
            "shape",
            "dtype",
            "channels",
            "min_value",
            "max_value",
            "mean_value",
            "std_deviation",
        ]
        for key in expected_keys:
            assert key in properties

        # Check specific property values
        assert properties["shape"] == (200, 200, 3)
        assert properties["channels"] == 3
        assert properties["min_value"] == 0
        assert properties["max_value"] == 255

        # Test with grayscale image
        gray_image = processor.preprocess(image)
        gray_properties = processor.analyze_image_properties(gray_image)

        assert gray_properties["channels"] == 1
        assert gray_properties["shape"] == (200, 200)

    def test_save_debug_image(self, test_image_path, default_config):
        """Test saving debug images."""
        processor = ImageProcessor(default_config)
        image = processor.load_image(test_image_path)

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir)
            processor.save_debug_image(image, output_path, prefix="test")

            # Check that a file was created
            files = list(output_path.glob("test_*.png"))
            assert len(files) == 1

            # Verify the saved image can be loaded
            saved_image = cv2.imread(str(files[0]))
            assert saved_image is not None
            assert saved_image.shape == (200, 200, 3)

    def test_background_leveling(self, default_config):
        """Test the background leveling feature."""
        # Create an image with a gradient background
        image = np.zeros((200, 200), dtype=np.uint8)
        for i in range(200):
            image[i, :] = i  # Gradient from 0 to 199

        # Add some "fibers" (white dots)
        for _ in range(100):
            x, y = np.random.randint(0, 200, 2)
            cv2.circle(image, (x, y), 2, (255, 255, 255), -1)

        # Convert to 3-channel image for preprocessing
        image_3c = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        # Config with leveling enabled
        leveling_config = ProcessingConfig()
        leveling_config.leveling.enabled = True
        leveling_config.leveling.kernel_size = 25
        processor = ImageProcessor(leveling_config)

        # Process the image
        processed_image = processor.preprocess(image_3c)

        # The background should be much more uniform (lower standard deviation)
        # We check the std dev of the non-fiber parts of the image
        # (where the original image was not 255)
        original_background_std = np.std(image[image < 255])
        processed_background_std = np.std(processed_image[image < 255])

        assert processed_background_std < original_background_std
