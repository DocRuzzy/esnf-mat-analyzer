"""
Main application module for nanofiber thickness uniformity analysis.

This module provides the entry point for the application and dependency injection setup.
"""

import argparse
import logging
from pathlib import Path
import yaml
import json
from typing import Dict, Any

# Import configuration and interfaces
from nanofiber_analyzer.config.config_manager import (
    Config,
    ProcessingConfig,
    CircleDetectionConfig,
    ThicknessConfig,
    UniformityConfig,
    VisualizationConfig,
)

# Import implementations
from nanofiber_analyzer.processing.image_processor import ImageProcessor
from nanofiber_analyzer.processing.circle_detector import CircleDetector
from nanofiber_analyzer.processing.thickness_estimator import ThicknessEstimator
from nanofiber_analyzer.analysis.radial_uniformity import RadialUniformityIndex
from nanofiber_analyzer.analysis.gini_coefficient import GiniCoefficient
from nanofiber_analyzer.analysis.thickness_ratio import ThicknessRangeRatio
from nanofiber_analyzer.visualization.visualizer import Visualizer
from nanofiber_analyzer.utils.data_exporter import DataExporter
from nanofiber_analyzer.core.analyzer import NanoFiberAnalyzer


def load_config(config_path: Path) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.

    Args:
        config_path: Path to the configuration file

    Returns:
        Dictionary containing configuration values
    """
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def create_config(config_data: Dict[str, Any] = None) -> Config:
    """
    Create configuration objects from dictionary data.

    Args:
        config_data: Dictionary of configuration values (optional)

    Returns:
        Config object
    """
    if config_data is None:
        config_data = {}

    # Create processing config
    processing_config = ProcessingConfig(
        blur_kernel_size=config_data.get("processing", {}).get("blur_kernel_size", 5),
        contrast_alpha=config_data.get("processing", {}).get("contrast_alpha", 1.5),
        contrast_beta=config_data.get("processing", {}).get("contrast_beta", 0),
        grayscale_conversion=config_data.get("processing", {}).get(
            "grayscale_conversion", "weighted"
        ),
    )

    # Create circle detection config
    circle_detection_config = CircleDetectionConfig(
        min_radius=config_data.get("circle_detection", {}).get("min_radius", 50),
        max_radius=config_data.get("circle_detection", {}).get("max_radius", 500),
        detection_method=config_data.get("circle_detection", {}).get(
            "detection_method", "hough"
        ),
        param1=config_data.get("circle_detection", {}).get("param1", 50),
        param2=config_data.get("circle_detection", {}).get("param2", 30),
    )

    # Create thickness config
    thickness_config = ThicknessConfig(
        model_type=config_data.get("thickness", {}).get("model_type", "linear"),
        a=config_data.get("thickness", {}).get("a", 1.0),
        b=config_data.get("thickness", {}).get("b", 0.0),
        saturation_threshold=config_data.get("thickness", {}).get(
            "saturation_threshold", 250
        ),
    )

    # Create uniformity config
    uniformity_config = UniformityConfig(
        num_radial_lines=config_data.get("uniformity", {}).get("num_radial_lines", 36),
        bin_count=config_data.get("uniformity", {}).get("bin_count", 50),
        smoothing_factor=config_data.get("uniformity", {}).get("smoothing_factor", 0.5),
    )

    # Create visualization config
    visualization_config = VisualizationConfig(
        colormap=config_data.get("visualization", {}).get("colormap", "viridis"),
        dpi=config_data.get("visualization", {}).get("dpi", 300),
        figure_size=config_data.get("visualization", {}).get("figure_size", (10, 8)),
        show_saturated=config_data.get("visualization", {}).get("show_saturated", True),
    )

    # Create main config
    config = Config(
        processing=processing_config,
        circle_detection=circle_detection_config,
        thickness=thickness_config,
        uniformity=uniformity_config,
        visualization=visualization_config,
        output_dir=Path(config_data.get("output_dir", "./output")),
        log_level=getattr(logging, config_data.get("log_level", "INFO")),
    )

    return config


def setup_dependencies(config: Config) -> NanoFiberAnalyzer:
    """
    Set up dependencies and create the main analyzer object.

    Args:
        config: Configuration object

    Returns:
        NanoFiberAnalyzer instance
    """
    # Create components
    image_processor = ImageProcessor(config.processing)
    circle_detector = CircleDetector(config.circle_detection)
    thickness_estimator = ThicknessEstimator(config.thickness)

    # Create uniformity metrics
    uniformity_metrics = [
        RadialUniformityIndex(config.uniformity),
        GiniCoefficient(),
        ThicknessRangeRatio(),
    ]

    # Create visualizer and data exporter
    visualizer = Visualizer(config.visualization)
    data_exporter = DataExporter()

    # Create main analyzer
    analyzer = NanoFiberAnalyzer(
        config=config,
        image_processor=image_processor,
        circle_detector=circle_detector,
        thickness_estimator=thickness_estimator,
        uniformity_metrics=uniformity_metrics,
        visualizer=visualizer,
        data_exporter=data_exporter,
    )

    return analyzer


def generate_default_config(output_path: Path) -> None:
    """
    Generate a default configuration file.

    Args:
        output_path: Path to write the configuration file
    """
    # Create default config
    config = create_config()

    # Convert to dictionary
    config_dict = {
        "processing": {
            "blur_kernel_size": config.processing.blur_kernel_size,
            "contrast_alpha": config.processing.contrast_alpha,
            "contrast_beta": config.processing.contrast_beta,
            "grayscale_conversion": config.processing.grayscale_conversion,
        },
        "circle_detection": {
            "min_radius": config.circle_detection.min_radius,
            "max_radius": config.circle_detection.max_radius,
            "detection_method": config.circle_detection.detection_method,
            "param1": config.circle_detection.param1,
            "param2": config.circle_detection.param2,
        },
        "thickness": {
            "model_type": config.thickness.model_type,
            "a": config.thickness.a,
            "b": config.thickness.b,
            "saturation_threshold": config.thickness.saturation_threshold,
        },
        "uniformity": {
            "num_radial_lines": config.uniformity.num_radial_lines,
            "bin_count": config.uniformity.bin_count,
            "smoothing_factor": config.uniformity.smoothing_factor,
        },
        "visualization": {
            "colormap": config.visualization.colormap,
            "dpi": config.visualization.dpi,
            "figure_size": config.visualization.figure_size,
            "show_saturated": config.visualization.show_saturated,
        },
        "output_dir": str(config.output_dir),
        "log_level": logging.getLevelName(config.log_level),
    }

    # Write to YAML file
    with open(output_path, "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False)


def main():
    """Main entry point for the application."""
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Analyze nanofiber thickness uniformity from images."
    )

    # Define arguments
    parser.add_argument(
        "--input", "-i", type=str, required=True, help="Input image file or directory"
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="./output",
        help="Output directory for results",
    )
    parser.add_argument("--config", "-c", type=str, help="Path to configuration file")
    parser.add_argument(
        "--pattern",
        "-p",
        type=str,
        default="*.jpg",
        help="File pattern for batch processing (when input is a directory)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )
    parser.add_argument(
        "--generate-config",
        type=str,
        help="Generate a default configuration file at the specified path",
    )

    # Parse arguments
    args = parser.parse_args()

    # Generate default config if requested
    if args.generate_config:
        generate_default_config(Path(args.generate_config))
        print(f"Default configuration generated at {args.generate_config}")
        return 0

    # Load configuration
    config_data = {}
    if args.config:
        config_path = Path(args.config)
        if config_path.exists():
            config_data = load_config(config_path)
        else:
            print(f"Warning: Configuration file not found: {config_path}")

    # Create config object
    config = create_config(config_data)

    # Override with command-line arguments
    config.output_dir = Path(args.output)
    if args.verbose:
        config.log_level = logging.DEBUG

    # Set up dependencies
    analyzer = setup_dependencies(config)

    # Process input
    input_path = Path(args.input)
    if input_path.is_file():
        # Process single file
        try:
            result = analyzer.process_image(input_path)
            print(f"Analysis complete. Results saved to {result['results_dir']}")
        except Exception as e:
            print(f"Error processing image {input_path}: {e}")
            return 1
    elif input_path.is_dir():
        # Process directory
        try:
            results = analyzer.batch_process(input_path, args.pattern)
            print(f"Batch processing complete. {len(results)} images processed.")

            # Save batch results summary
            summary_path = Path(args.output) / "batch_summary.json"
            with open(summary_path, "w") as f:
                json.dump(results, f, indent=2)

            print(f"Batch summary saved to {summary_path}")
        except Exception as e:
            print(f"Error during batch processing: {e}")
            return 1
    else:
        print(f"Error: Input path does not exist: {input_path}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
