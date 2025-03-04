"""
Configuration management for the ESNF Mat Analyzer.

This module provides functions for loading, saving, and validating configuration
settings. It handles conversion between the data structures defined in data_types.py
and serializable formats (YAML, JSON).
"""

import yaml
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, cast
import argparse

from esnf_mat_analyzer.core.data_types import (
    Config,
    ProcessingConfig,
    CircleDetectionConfig,
    ThicknessConfig,
    UniformityConfig,
    VisualizationConfig,
    ExportConfig,
    GrayscaleConversionMethod,
    CircleDetectionMethod,
    ThicknessModelType,
)

# Setup logging
logger = logging.getLogger(__name__)

# Enum conversion maps
GRAYSCALE_METHOD_MAP = {
    "weighted": GrayscaleConversionMethod.WEIGHTED,
    "average": GrayscaleConversionMethod.AVERAGE,
    "luminance": GrayscaleConversionMethod.LUMINANCE,
}

CIRCLE_DETECTION_METHOD_MAP = {
    "hough": CircleDetectionMethod.HOUGH,
    "contour": CircleDetectionMethod.CONTOUR,
}

THICKNESS_MODEL_MAP = {
    "linear": ThicknessModelType.LINEAR,
    "logarithmic": ThicknessModelType.LOGARITHMIC,
    "exponential": ThicknessModelType.EXPONENTIAL,
}


def enum_to_str(enum_value: Any) -> str:
    """
    Convert an enum value to its string representation.

    Args:
        enum_value: Enum value to convert

    Returns:
        String representation of the enum value
    """
    if isinstance(enum_value, GrayscaleConversionMethod):
        return {
            GrayscaleConversionMethod.WEIGHTED: "weighted",
            GrayscaleConversionMethod.AVERAGE: "average",
            GrayscaleConversionMethod.LUMINANCE: "luminance",
        }[enum_value]
    elif isinstance(enum_value, CircleDetectionMethod):
        return {
            CircleDetectionMethod.HOUGH: "hough",
            CircleDetectionMethod.CONTOUR: "contour",
        }[enum_value]
    elif isinstance(enum_value, ThicknessModelType):
        return {
            ThicknessModelType.LINEAR: "linear",
            ThicknessModelType.LOGARITHMIC: "logarithmic",
            ThicknessModelType.EXPONENTIAL: "exponential",
        }[enum_value]
    else:
        return str(enum_value)


def str_to_enum(string_value: str, enum_type: Any) -> Any:
    """
    Convert a string to its corresponding enum value.

    Args:
        string_value: String to convert
        enum_type: Type of enum to convert to

    Returns:
        Enum value

    Raises:
        ValueError: If the string cannot be converted to the enum type
    """
    if enum_type == GrayscaleConversionMethod:
        if string_value.lower() not in GRAYSCALE_METHOD_MAP:
            valid_values = ", ".join(GRAYSCALE_METHOD_MAP.keys())
            raise ValueError(
                f"Invalid grayscale conversion method: {string_value}. "
                f"Valid values are: {valid_values}"
            )
        return GRAYSCALE_METHOD_MAP[string_value.lower()]
    elif enum_type == CircleDetectionMethod:
        if string_value.lower() not in CIRCLE_DETECTION_METHOD_MAP:
            valid_values = ", ".join(CIRCLE_DETECTION_METHOD_MAP.keys())
            raise ValueError(
                f"Invalid circle detection method: {string_value}. "
                f"Valid values are: {valid_values}"
            )
        return CIRCLE_DETECTION_METHOD_MAP[string_value.lower()]
    elif enum_type == ThicknessModelType:
        if string_value.lower() not in THICKNESS_MODEL_MAP:
            valid_values = ", ".join(THICKNESS_MODEL_MAP.keys())
            raise ValueError(
                f"Invalid thickness model type: {string_value}. "
                f"Valid values are: {valid_values}"
            )
        return THICKNESS_MODEL_MAP[string_value.lower()]
    else:
        raise TypeError(f"Unsupported enum type: {enum_type}")


def config_to_dict(config: Config) -> Dict[str, Any]:
    """
    Convert a Config object to a dictionary suitable for serialization.

    Args:
        config: Config object to convert

    Returns:
        Dictionary representation of the Config object
    """
    # Convert enums to strings
    processing_dict = {
        "blur_kernel_size": config.processing.blur_kernel_size,
        "contrast_alpha": config.processing.contrast_alpha,
        "contrast_beta": config.processing.contrast_beta,
        "grayscale_conversion": enum_to_str(config.processing.grayscale_conversion),
    }

    circle_detection_dict = {
        "min_radius": config.circle_detection.min_radius,
        "max_radius": config.circle_detection.max_radius,
        "detection_method": enum_to_str(config.circle_detection.detection_method),
        "param1": config.circle_detection.param1,
        "param2": config.circle_detection.param2,
        "min_area": config.circle_detection.min_area,
        "max_area": config.circle_detection.max_area,
    }

    thickness_dict = {
        "model_type": enum_to_str(config.thickness.model_type),
        "a": config.thickness.a,
        "b": config.thickness.b,
        "saturation_threshold": config.thickness.saturation_threshold,
        "normalization": config.thickness.normalization,
        "calibration_factor": config.thickness.calibration_factor,
    }

    uniformity_dict = {
        "num_radial_lines": config.uniformity.num_radial_lines,
        "bin_count": config.uniformity.bin_count,
        "smoothing_factor": config.uniformity.smoothing_factor,
        "min_thickness_percentile": config.uniformity.min_thickness_percentile,
        "max_thickness_percentile": config.uniformity.max_thickness_percentile,
        "ignore_saturated": config.uniformity.ignore_saturated,
    }

    visualization_dict = {
        "colormap": config.visualization.colormap,
        "dpi": config.visualization.dpi,
        "figure_size": list(config.visualization.figure_size),
        "show_saturated": config.visualization.show_saturated,
        "radial_avg_line_color": config.visualization.radial_avg_line_color,
        "radial_std_fill_color": config.visualization.radial_std_fill_color,
        "histogram_color": config.visualization.histogram_color,
        "histogram_edge_color": config.visualization.histogram_edge_color,
    }

    export_dict = {
        "csv_delimiter": config.export.csv_delimiter,
        "export_formats": config.export.export_formats,
        "compress_outputs": config.export.compress_outputs,
        "include_metadata": config.export.include_metadata,
    }

    # Combine into main config dictionary
    config_dict = {
        "processing": processing_dict,
        "circle_detection": circle_detection_dict,
        "thickness": thickness_dict,
        "uniformity": uniformity_dict,
        "visualization": visualization_dict,
        "export": export_dict,
        "output_dir": str(config.output_dir),
        "log_level": logging.getLevelName(config.log_level),
        "cache_intermediates": config.cache_intermediates,
    }

    return config_dict


def dict_to_config(config_dict: Dict[str, Any]) -> Config:
    """
    Convert a dictionary to a Config object.

    Args:
        config_dict: Dictionary to convert

    Returns:
        Config object

    Raises:
        ValueError: If the dictionary contains invalid values
    """
    # Create individual config components with default values
    processing_config = ProcessingConfig()
    circle_detection_config = CircleDetectionConfig()
    thickness_config = ThicknessConfig()
    uniformity_config = UniformityConfig()
    visualization_config = VisualizationConfig()
    export_config = ExportConfig()

    # Update with values from dictionary, if present
    if "processing" in config_dict:
        proc_dict = config_dict["processing"]
        if "blur_kernel_size" in proc_dict:
            processing_config.blur_kernel_size = int(proc_dict["blur_kernel_size"])
        if "contrast_alpha" in proc_dict:
            processing_config.contrast_alpha = float(proc_dict["contrast_alpha"])
        if "contrast_beta" in proc_dict:
            processing_config.contrast_beta = int(proc_dict["contrast_beta"])
        if "grayscale_conversion" in proc_dict:
            processing_config.grayscale_conversion = str_to_enum(
                proc_dict["grayscale_conversion"], GrayscaleConversionMethod
            )

    if "circle_detection" in config_dict:
        circ_dict = config_dict["circle_detection"]
        if "min_radius" in circ_dict:
            circle_detection_config.min_radius = int(circ_dict["min_radius"])
        if "max_radius" in circ_dict:
            circle_detection_config.max_radius = int(circ_dict["max_radius"])
        if "detection_method" in circ_dict:
            circle_detection_config.detection_method = str_to_enum(
                circ_dict["detection_method"], CircleDetectionMethod
            )
        if "param1" in circ_dict:
            circle_detection_config.param1 = int(circ_dict["param1"])
        if "param2" in circ_dict:
            circle_detection_config.param2 = int(circ_dict["param2"])
        if "min_area" in circ_dict:
            circle_detection_config.min_area = int(circ_dict["min_area"])
        if "max_area" in circ_dict:
            circle_detection_config.max_area = int(circ_dict["max_area"])

    if "thickness" in config_dict:
        thick_dict = config_dict["thickness"]
        if "model_type" in thick_dict:
            thickness_config.model_type = str_to_enum(
                thick_dict["model_type"], ThicknessModelType
            )
        if "a" in thick_dict:
            thickness_config.a = float(thick_dict["a"])
        if "b" in thick_dict:
            thickness_config.b = float(thick_dict["b"])
        if "saturation_threshold" in thick_dict:
            thickness_config.saturation_threshold = int(
                thick_dict["saturation_threshold"]
            )
        if "normalization" in thick_dict:
            thickness_config.normalization = bool(thick_dict["normalization"])
        if (
            "calibration_factor" in thick_dict
            and thick_dict["calibration_factor"] is not None
        ):
            thickness_config.calibration_factor = float(
                thick_dict["calibration_factor"]
            )

    if "uniformity" in config_dict:
        unif_dict = config_dict["uniformity"]
        if "num_radial_lines" in unif_dict:
            uniformity_config.num_radial_lines = int(unif_dict["num_radial_lines"])
        if "bin_count" in unif_dict:
            uniformity_config.bin_count = int(unif_dict["bin_count"])
        if "smoothing_factor" in unif_dict:
            uniformity_config.smoothing_factor = float(unif_dict["smoothing_factor"])
        if "min_thickness_percentile" in unif_dict:
            uniformity_config.min_thickness_percentile = float(
                unif_dict["min_thickness_percentile"]
            )
        if "max_thickness_percentile" in unif_dict:
            uniformity_config.max_thickness_percentile = float(
                unif_dict["max_thickness_percentile"]
            )
        if "ignore_saturated" in unif_dict:
            uniformity_config.ignore_saturated = bool(unif_dict["ignore_saturated"])

    if "visualization" in config_dict:
        vis_dict = config_dict["visualization"]
        if "colormap" in vis_dict:
            visualization_config.colormap = str(vis_dict["colormap"])
        if "dpi" in vis_dict:
            visualization_config.dpi = int(vis_dict["dpi"])
        if "figure_size" in vis_dict:
            fig_size = vis_dict["figure_size"]
            if isinstance(fig_size, (list, tuple)) and len(fig_size) == 2:
                visualization_config.figure_size = (int(fig_size[0]), int(fig_size[1]))
        if "show_saturated" in vis_dict:
            visualization_config.show_saturated = bool(vis_dict["show_saturated"])
        if "radial_avg_line_color" in vis_dict:
            visualization_config.radial_avg_line_color = str(
                vis_dict["radial_avg_line_color"]
            )
        if "radial_std_fill_color" in vis_dict:
            visualization_config.radial_std_fill_color = str(
                vis_dict["radial_std_fill_color"]
            )
        if "histogram_color" in vis_dict:
            visualization_config.histogram_color = str(vis_dict["histogram_color"])
        if "histogram_edge_color" in vis_dict:
            visualization_config.histogram_edge_color = str(
                vis_dict["histogram_edge_color"]
            )

    if "export" in config_dict:
        exp_dict = config_dict["export"]
        if "csv_delimiter" in exp_dict:
            export_config.csv_delimiter = str(exp_dict["csv_delimiter"])
        if "export_formats" in exp_dict:
            export_config.export_formats = list(exp_dict["export_formats"])
        if "compress_outputs" in exp_dict:
            export_config.compress_outputs = bool(exp_dict["compress_outputs"])
        if "include_metadata" in exp_dict:
            export_config.include_metadata = bool(exp_dict["include_metadata"])

    # Create output directory Path
    output_dir = Path("./output")
    if "output_dir" in config_dict:
        output_dir = Path(config_dict["output_dir"])

    # Parse log level
    log_level = logging.INFO
    if "log_level" in config_dict:
        level_name = config_dict["log_level"]
        numeric_level = getattr(logging, level_name, None)
        if isinstance(numeric_level, int):
            log_level = numeric_level
        else:
            logger.warning(f"Invalid log level: {level_name}, using INFO instead")

    # Parse cache_intermediates
    cache_intermediates = True
    if "cache_intermediates" in config_dict:
        cache_intermediates = bool(config_dict["cache_intermediates"])

    # Create main Config object
    config = Config(
        processing=processing_config,
        circle_detection=circle_detection_config,
        thickness=thickness_config,
        uniformity=uniformity_config,
        visualization=visualization_config,
        export=export_config,
        output_dir=output_dir,
        log_level=log_level,
        cache_intermediates=cache_intermediates,
    )

    return config


def load_config(config_path: Path) -> Config:
    """
    Load configuration from a YAML file.

    Args:
        config_path: Path to the configuration file

    Returns:
        Config object

    Raises:
        FileNotFoundError: If the configuration file does not exist
        ValueError: If the configuration file contains invalid values
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    logger.info(f"Loading configuration from {config_path}")

    try:
        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f)

        if config_dict is None:
            logger.warning(f"Empty configuration file: {config_path}, using defaults")
            return Config()

        config = dict_to_config(config_dict)
        logger.info("Configuration loaded successfully")
        return config

    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML config: {e}")
        raise ValueError(f"Invalid YAML in configuration file: {e}")

    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        raise


def save_config(config: Config, config_path: Path) -> None:
    """
    Save configuration to a YAML file.

    Args:
        config: Config object to save
        config_path: Path to save the configuration to

    Raises:
        IOError: If the configuration file cannot be written
    """
    logger.info(f"Saving configuration to {config_path}")

    # Create parent directory if it doesn't exist
    config_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        config_dict = config_to_dict(config)

        with open(config_path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

        logger.info("Configuration saved successfully")

    except Exception as e:
        logger.error(f"Error saving configuration: {e}")
        raise IOError(f"Failed to save configuration: {e}")


def validate_config(config: Config) -> List[str]:
    """
    Validate a Config object.

    Args:
        config: Config object to validate

    Returns:
        List of validation error messages (empty if valid)
    """
    errors = []

    # Validate ProcessingConfig
    if config.processing.blur_kernel_size < 0:
        errors.append("blur_kernel_size must be non-negative")
    elif (
        config.processing.blur_kernel_size > 0
        and config.processing.blur_kernel_size % 2 == 0
    ):
        errors.append("blur_kernel_size must be odd when non-zero")

    if config.processing.contrast_alpha <= 0:
        errors.append("contrast_alpha must be positive")

    if config.processing.contrast_beta < -255 or config.processing.contrast_beta > 255:
        errors.append("contrast_beta must be between -255 and 255")

    # Validate CircleDetectionConfig
    if config.circle_detection.min_radius <= 0:
        errors.append("min_radius must be positive")

    if config.circle_detection.max_radius <= config.circle_detection.min_radius:
        errors.append("max_radius must be greater than min_radius")

    if config.circle_detection.param1 <= 0:
        errors.append("param1 must be positive")

    if config.circle_detection.param2 <= 0:
        errors.append("param2 must be positive")

    if config.circle_detection.min_area <= 0:
        errors.append("min_area must be positive")

    if config.circle_detection.max_area <= config.circle_detection.min_area:
        errors.append("max_area must be greater than min_area")

    # Validate ThicknessConfig
    if (
        config.thickness.saturation_threshold <= 0
        or config.thickness.saturation_threshold > 255
    ):
        errors.append("saturation_threshold must be between 1 and 255")

    # Validate UniformityConfig
    if config.uniformity.num_radial_lines <= 0:
        errors.append("num_radial_lines must be positive")

    if config.uniformity.bin_count <= 0:
        errors.append("bin_count must be positive")

    if config.uniformity.smoothing_factor < 0 or config.uniformity.smoothing_factor > 1:
        errors.append("smoothing_factor must be between 0 and 1")

    if (
        config.uniformity.min_thickness_percentile < 0
        or config.uniformity.min_thickness_percentile > 100
    ):
        errors.append("min_thickness_percentile must be between 0 and 100")

    if (
        config.uniformity.max_thickness_percentile < 0
        or config.uniformity.max_thickness_percentile > 100
    ):
        errors.append("max_thickness_percentile must be between 0 and 100")

    if (
        config.uniformity.min_thickness_percentile
        >= config.uniformity.max_thickness_percentile
    ):
        errors.append(
            "min_thickness_percentile must be less than max_thickness_percentile"
        )

    # Validate VisualizationConfig
    if config.visualization.dpi <= 0:
        errors.append("dpi must be positive")

    if (
        config.visualization.figure_size[0] <= 0
        or config.visualization.figure_size[1] <= 0
    ):
        errors.append("figure_size dimensions must be positive")

    return errors


def get_default_config() -> Config:
    """
    Get the default configuration.

    Returns:
        Default Config object
    """
    return Config()


def generate_default_config_file(config_path: Path) -> None:
    """
    Generate a default configuration file.

    Args:
        config_path: Path to save the default configuration to

    Raises:
        IOError: If the configuration file cannot be written
    """
    default_config = get_default_config()
    save_config(default_config, config_path)
    logger.info(f"Default configuration generated at {config_path}")


def parse_config_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    """
    Parse command-line arguments related to configuration.

    Args:
        args: Command-line arguments (None to use sys.argv)

    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description="ESNF Mat Analyzer Configuration")

    parser.add_argument("--config", "-c", type=str, help="Path to configuration file")

    parser.add_argument(
        "--generate-config",
        type=str,
        help="Generate a default configuration file at the specified path",
    )

    parser.add_argument(
        "--output-dir", "-o", type=str, help="Output directory for analysis results"
    )

    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level",
    )

    return parser.parse_args(args)


def update_config_from_args(config: Config, args: argparse.Namespace) -> Config:
    """
    Update a Config object with values from command-line arguments.

    Args:
        config: Config object to update
        args: Parsed command-line arguments

    Returns:
        Updated Config object
    """
    if args.output_dir:
        config.output_dir = Path(args.output_dir)

    if args.log_level:
        config.log_level = getattr(logging, args.log_level)

    return config


def load_config_with_args(args: Optional[argparse.Namespace] = None) -> Config:
    """
    Load configuration from file and command-line arguments.

    Args:
        args: Parsed command-line arguments (None to parse from sys.argv)

    Returns:
        Config object

    Raises:
        FileNotFoundError: If the configuration file does not exist
        ValueError: If the configuration contains invalid values
    """
    if args is None:
        args = parse_config_args()

    # Generate default config if requested
    if args.generate_config:
        generate_default_config_file(Path(args.generate_config))
        logger.info(f"Default configuration generated at {args.generate_config}")

    # Load config from file if specified
    if args.config:
        config = load_config(Path(args.config))
    else:
        config = get_default_config()
        logger.info("No configuration file specified, using defaults")

    # Update with command-line arguments
    config = update_config_from_args(config, args)

    # Validate the configuration
    errors = validate_config(config)
    if errors:
        error_msg = "\n".join(errors)
        logger.error(f"Invalid configuration:\n{error_msg}")
        raise ValueError(f"Invalid configuration:\n{error_msg}")

    return config
