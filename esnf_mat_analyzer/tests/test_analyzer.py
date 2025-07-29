import pytest
from unittest.mock import MagicMock, patch, call
from pathlib import Path
import numpy as np # For dummy image data if needed by mocks

# Classes to be tested or used in type hints
from esnf_mat_analyzer.core.analyzer import NanoFiberAnalyzer
from esnf_mat_analyzer.core.data_types import (
    Config, AnalysisResult, Point, ProcessingConfig, CircleDetectionConfig, 
    ThicknessConfig, UniformityConfig, VisualizationConfig, ExportConfig, 
    RulerDetectionConfig
)
# Interfaces for mocking
from esnf_mat_analyzer.core.interfaces import (
    ImageProcessorInterface, CircleDetectorInterface, ThicknessEstimatorInterface,
    UniformityMetricInterface, VisualizerInterface, DataExporterInterface
)
from esnf_mat_analyzer.processing.ruler_detector import RulerDetector


@pytest.fixture
def mock_config() -> Config:
    """Returns a default mock Config object."""
    return Config(
        processing=ProcessingConfig(),
        circle_detection=CircleDetectionConfig(),
        thickness=ThicknessConfig(),
        uniformity=UniformityConfig(),
        visualization=VisualizationConfig(),
        export=ExportConfig(),
        ruler_detection=RulerDetectionConfig(enabled=True), # Ruler detection enabled by default for some tests
        output_dir=Path("./test_output"),
        log_level="DEBUG" 
    )

@pytest.fixture
def mock_image_processor() -> MagicMock:
    mock = MagicMock(spec=ImageProcessorInterface)
    # Configure return values for methods that produce data for the pipeline
    mock.load_image.return_value = np.zeros((100, 100, 3), dtype=np.uint8) # Dummy RGB image
    mock.preprocess.return_value = np.zeros((100, 100), dtype=np.uint8)   # Dummy Grayscale image
    return mock

@pytest.fixture
def mock_circle_detector() -> MagicMock:
    mock = MagicMock(spec=CircleDetectorInterface)
    mock.detect.return_value = ((50, 50), 40) # center (Point), radius (int)
    mock.create_mask.return_value = np.ones((100, 100), dtype=np.uint8) # Dummy mask
    return mock

@pytest.fixture
def mock_thickness_estimator() -> MagicMock:
    mock = MagicMock(spec=ThicknessEstimatorInterface)
    mock.estimate.return_value = np.random.rand(100, 100).astype(np.float32) # Dummy thickness map
    mock.get_saturation_mask.return_value = np.zeros((100, 100), dtype=np.uint8) # Dummy saturation mask
    return mock

@pytest.fixture
def mock_ruler_detector() -> MagicMock:
    mock = MagicMock(spec=RulerDetector) # Using concrete class for spec if methods are specific
    mock.detect_scale.return_value = 10.0 # pixels/mm
    return mock

@pytest.fixture
def mock_uniformity_metric() -> MagicMock:
    mock = MagicMock(spec=UniformityMetricInterface)
    mock.name = "MockUniformity"
    mock.calculate.return_value = 0.75
    return mock

@pytest.fixture
def mock_visualizer() -> MagicMock:
    return MagicMock(spec=VisualizerInterface)

@pytest.fixture
def mock_data_exporter() -> MagicMock:
    return MagicMock(spec=DataExporterInterface)

@pytest.fixture
def analyzer(mock_config, mock_image_processor, mock_circle_detector, 
             mock_thickness_estimator, mock_ruler_detector, 
             mock_uniformity_metric, mock_visualizer, mock_data_exporter) -> NanoFiberAnalyzer:
    """Fixture to create NanoFiberAnalyzer with mocked dependencies."""
    return NanoFiberAnalyzer(
        config=mock_config,
        image_processor=mock_image_processor,
        circle_detector=mock_circle_detector,
        thickness_estimator=mock_thickness_estimator,
        ruler_detector=mock_ruler_detector,
        uniformity_metrics=[mock_uniformity_metric],
        visualizer=mock_visualizer,
        data_exporter=mock_data_exporter
    )

class TestNanoFiberAnalyzer:

    def test_initialization(self, analyzer: NanoFiberAnalyzer, mock_config: Config):
        """Test if NanoFiberAnalyzer initializes correctly."""
        assert analyzer.config == mock_config
        assert analyzer.logger is not None

    def test_process_image_successful_flow(self, analyzer: NanoFiberAnalyzer, mock_image_processor: MagicMock, 
                                           mock_ruler_detector: MagicMock, mock_circle_detector: MagicMock,
                                           mock_thickness_estimator: MagicMock, mock_uniformity_metric: MagicMock):
        """Test the successful processing flow of a single image."""
        test_image_path = Path("dummy_image.png")
        
        result = analyzer.process_image(test_image_path)

        # Check if core methods of dependencies were called
        mock_image_processor.load_image.assert_called_once_with(test_image_path)
        mock_ruler_detector.detect_scale.assert_called_once_with(mock_image_processor.load_image.return_value)
        mock_image_processor.preprocess.assert_called_once_with(mock_image_processor.load_image.return_value)
        mock_circle_detector.detect.assert_called_once_with(mock_image_processor.preprocess.return_value)
        mock_circle_detector.create_mask.assert_called_once()
        mock_thickness_estimator.estimate.assert_called_once()
        mock_thickness_estimator.get_saturation_mask.assert_called_once()
        mock_uniformity_metric.calculate.assert_called_once()

        # Check results
        assert isinstance(result, AnalysisResult)
        assert result.image_path == test_image_path
        assert result.metrics[mock_uniformity_metric.name] == mock_uniformity_metric.calculate.return_value
        assert result.spatial_scale_pixels_per_mm == mock_ruler_detector.detect_scale.return_value
        assert result.center == mock_circle_detector.detect.return_value[0]
        assert result.radius == mock_circle_detector.detect.return_value[1]

    def test_process_image_ruler_detection_disabled(self, analyzer: NanoFiberAnalyzer, mock_config: Config, 
                                                     mock_ruler_detector: MagicMock):
        """Test that ruler detection is skipped if disabled in config."""
        mock_config.ruler_detection.enabled = False
        # Re-initialize analyzer or ensure config is mutable and affects behavior
        # For this test, assume config is mutable or re-initialize for clarity if needed.
        # If analyzer takes a deepcopy of config, this won't work without re-init.
        # Let's assume direct use for now. If not, test needs analyzer re-init.
        
        test_image_path = Path("dummy_image_disabled_ruler.png")
        analyzer.process_image(test_image_path)
        
        mock_ruler_detector.detect_scale.assert_not_called()
        
        # Reset for other tests if config is shared by fixture scope
        mock_config.ruler_detection.enabled = True 

    def test_process_image_load_failure(self, analyzer: NanoFiberAnalyzer, mock_image_processor: MagicMock):
        """Test handling of image load failure."""
        test_image_path = Path("non_existent.png")
        mock_image_processor.load_image.side_effect = FileNotFoundError("File not found")
        
        with pytest.raises(FileNotFoundError):
            analyzer.process_image(test_image_path)

    def test_process_image_ruler_detection_exception(self, analyzer: NanoFiberAnalyzer, mock_ruler_detector: MagicMock, caplog):
        """Test handling of an exception during ruler detection."""
        test_image_path = Path("ruler_exception.png")
        mock_ruler_detector.detect_scale.side_effect = Exception("Ruler crash")
        
        # The processing should continue, but log an error and scale should be None
        result = analyzer.process_image(test_image_path)
        
        assert "Error during scale detection: Ruler crash" in caplog.text
        assert result.spatial_scale_pixels_per_mm is None


    @patch('esnf_mat_analyzer.core.analyzer.Path.glob') # Patch glob where it's used
    def test_batch_process_successful(self, mock_glob: MagicMock, analyzer: NanoFiberAnalyzer, mock_config: Config):
        """Test successful batch processing of multiple files."""
        # Setup mock_glob to return a list of Path objects
        mock_image_paths = [Path("img1.png"), Path("img2.jpg")]
        mock_glob.return_value = mock_image_paths
        
        # Mock analyzer.process_image to simplify testing batch logic
        analyzer.process_image = MagicMock(return_value=AnalysisResult(
            image_path=Path("dummy"), center=(0,0), radius=0, 
            thickness_map=np.array([]), mask=np.array([]),
            spatial_scale_pixels_per_mm=5.0, # <<< Add here
            metadata={} 
        ))
        
        batch_results = analyzer.batch_process(mock_config.output_dir, "*.png")
        
        mock_glob.assert_called_once_with("*.png")
        assert analyzer.process_image.call_count == len(mock_image_paths)
        analyzer.process_image.assert_any_call(mock_image_paths[0])
        analyzer.process_image.assert_any_call(mock_image_paths[1])
        assert len(batch_results) == len(mock_image_paths)
        assert batch_results[0]['status'] == 'success'

    @patch('esnf_mat_analyzer.core.analyzer.Path.glob')
    def test_batch_process_no_images_found(self, mock_glob: MagicMock, analyzer: NanoFiberAnalyzer, mock_config: Config):
        """Test batch processing when no images match the pattern."""
        mock_glob.return_value = [] # No images found
        
        batch_results = analyzer.batch_process(mock_config.output_dir, "*.tif")
        
        mock_glob.assert_called_once_with("*.tif")
        assert len(batch_results) == 0

    @patch('esnf_mat_analyzer.core.analyzer.Path.glob')
    def test_batch_process_with_one_error(self, mock_glob: MagicMock, analyzer: NanoFiberAnalyzer, mock_config: Config):
        """Test batch processing where one image fails and others succeed."""
        mock_image_paths = [Path("img_ok.png"), Path("img_bad.jpg"), Path("img_ok2.png")]
        mock_glob.return_value = mock_image_paths

        # Simulate process_image: success for some, error for "img_bad.jpg"
        def process_image_side_effect(image_path):
            if image_path.name == "img_bad.jpg":
                raise ValueError("Simulated processing error")
            return AnalysisResult(
                image_path=image_path, center=(1,1), radius=1, 
                thickness_map=np.array([1]), mask=np.array([1]),
                spatial_scale_pixels_per_mm=1.0, # <<< Add here
                metadata={}
            )

        analyzer.process_image = MagicMock(side_effect=process_image_side_effect)
        
        batch_results = analyzer.batch_process(mock_config.output_dir, "*.png") # Pattern doesn't matter due to mock_glob
        
        assert analyzer.process_image.call_count == len(mock_image_paths)
        assert len(batch_results) == len(mock_image_paths)
        
        assert batch_results[0]['status'] == 'success'
        assert batch_results[1]['status'] == 'error'
        assert batch_results[1]['error_message'] == "Simulated processing error"
        assert batch_results[1]['image_path'] == str(mock_image_paths[1])
        assert batch_results[2]['status'] == 'success'
