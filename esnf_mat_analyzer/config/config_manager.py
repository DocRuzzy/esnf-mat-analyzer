
"""
Configuration management for the ESNF Mat Analyzer.

Author: ESNF Mat Analyzer Team
License: GNU General Public License v3.0 or later (GPLv3)

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
    ThicknessConfig,
    UniformityConfig,
    VisualizationConfig,
    ExportConfig,
    GrayscaleConversionMethod,
    ThicknessModelType,
    BackgroundCorrectionMethod,
)

# Setup logging
logger = logging.getLogger(__name__)

# Enum conversion maps
GRAYSCALE_METHOD_MAP = {
    "weighted": GrayscaleConversionMethod.WEIGHTED,
    "average": GrayscaleConversionMethod.AVERAGE,
    "luminance": GrayscaleConversionMethod.LUMINANCE,
}

THICKNESS_MODEL_MAP = {
    "linear": ThicknessModelType.LINEAR,
    "logarithmic": ThicknessModelType.LOGARITHMIC,
    "exponential": ThicknessModelType.EXPONENTIAL,
    "beer_lambert": ThicknessModelType.BEER_LAMBERT,
}

BACKGROUND_CORRECTION_METHOD_MAP = {
    "none": BackgroundCorrectionMethod.NONE,
    "basic": BackgroundCorrectionMethod.BASIC,
    "rolling_ball": BackgroundCorrectionMethod.ROLLING_BALL,
    "restore": BackgroundCorrectionMethod.RESTORE,
    "homomorphic": BackgroundCorrectionMethod.HOMOMORPHIC,
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
    elif isinstance(enum_value, ThicknessModelType):
        return {
            ThicknessModelType.LINEAR: "linear",
            ThicknessModelType.LOGARITHMIC: "logarithmic",
            ThicknessModelType.EXPONENTIAL: "exponential",
            ThicknessModelType.BEER_LAMBERT: "beer_lambert",
        }[enum_value]
    elif isinstance(enum_value, BackgroundCorrectionMethod):
        return {
            BackgroundCorrectionMethod.NONE: "none",
            BackgroundCorrectionMethod.BASIC: "basic",
            BackgroundCorrectionMethod.ROLLING_BALL: "rolling_ball",
            BackgroundCorrectionMethod.RESTORE: "restore",
            BackgroundCorrectionMethod.HOMOMORPHIC: "homomorphic",
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
    elif enum_type == ThicknessModelType:
        if string_value.lower() not in THICKNESS_MODEL_MAP:
            valid_values = ", ".join(THICKNESS_MODEL_MAP.keys())
            raise ValueError(
                f"Invalid thickness model type: {string_value}. "
                f"Valid values are: {valid_values}"
            )
        return THICKNESS_MODEL_MAP[string_value.lower()]
    elif enum_type == BackgroundCorrectionMethod:
        if string_value.lower() not in BACKGROUND_CORRECTION_METHOD_MAP:
            valid_values = ", ".join(BACKGROUND_CORRECTION_METHOD_MAP.keys())
            raise ValueError(
                f"Invalid background correction method: {string_value}. "
                f"Valid values are: {valid_values}"
            )
        return BACKGROUND_CORRECTION_METHOD_MAP[string_value.lower()]
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
        "background_correction_method": enum_to_str(config.processing.background_correction_method),
        "basic_correction_n_components": config.processing.basic_correction_n_components,
        "rolling_ball_radius": config.processing.rolling_ball_radius,
        "restore_percentile": config.processing.restore_percentile,
        "homomorphic_cutoff": config.processing.homomorphic_cutoff,
        "homomorphic_g_low": config.processing.homomorphic_g_low,
        "homomorphic_g_high": config.processing.homomorphic_g_high,
        "saturation_recovery": config.processing.saturation_recovery,
    }

    thickness_dict = {
        "model_type": enum_to_str(config.thickness.model_type),
        "a": config.thickness.a,
        "b": config.thickness.b,
        "saturation_threshold": config.thickness.saturation_threshold,
        "normalization": config.thickness.normalization,
        "calibration_factor": config.thickness.calibration_factor,
        "attenuation_coefficient": config.thickness.attenuation_coefficient,
    }

    uniformity_dict = {
        "num_radial_lines": config.uniformity.num_radial_lines,
        "bin_count": config.uniformity.bin_count,
        "smoothing_factor": config.uniformity.smoothing_factor,
        "min_thickness_percentile": config.uniformity.min_thickness_percentile,
        "max_thickness_percentile": config.uniformity.max_thickness_percentile,
        "ignore_saturated": config.uniformity.ignore_saturated,
        "glcm_distances": config.uniformity.glcm_distances,
        "glcm_angles": config.uniformity.glcm_angles,
        "lbp_radius": config.uniformity.lbp_radius,
        "lbp_points": config.uniformity.lbp_points,
        "max_coefficients": config.uniformity.max_coefficients,
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
        if "background_correction_method" in proc_dict:
            processing_config.background_correction_method = str_to_enum(
                proc_dict["background_correction_method"], BackgroundCorrectionMethod
            )
        if "basic_correction_n_components" in proc_dict:
            processing_config.basic_correction_n_components = int(proc_dict["basic_correction_n_components"])
        if "rolling_ball_radius" in proc_dict:
            processing_config.rolling_ball_radius = int(proc_dict["rolling_ball_radius"])
        if "restore_percentile" in proc_dict:
            processing_config.restore_percentile = float(proc_dict["restore_percentile"])
        if "homomorphic_cutoff" in proc_dict:
            processing_config.homomorphic_cutoff = float(proc_dict["homomorphic_cutoff"])
        if "homomorphic_g_low" in proc_dict:
            processing_config.homomorphic_g_low = float(proc_dict["homomorphic_g_low"])
        if "homomorphic_g_high" in proc_dict:
            processing_config.homomorphic_g_high = float(proc_dict["homomorphic_g_high"])
        if "saturation_recovery" in proc_dict:
            processing_config.saturation_recovery = bool(proc_dict["saturation_recovery"])

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
        if "attenuation_coefficient" in thick_dict:
            thickness_config.attenuation_coefficient = float(thick_dict["attenuation_coefficient"])

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
        if "glcm_distances" in unif_dict:
            uniformity_config.glcm_distances = list(unif_dict["glcm_distances"])
        if "glcm_angles" in unif_dict:
            uniformity_config.glcm_angles = list(unif_dict["glcm_angles"])
        if "lbp_radius" in unif_dict:
            uniformity_config.lbp_radius = int(unif_dict["lbp_radius"])
        if "lbp_points" in unif_dict:
            uniformity_config.lbp_points = int(unif_dict["lbp_points"])
        if "max_coefficients" in unif_dict:
            uniformity_config.max_coefficients = int(unif_dict["max_coefficients"])

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
    Comprehensively validate a configuration object for scientific analysis workflows.
    
    This function performs extensive validation of all configuration parameters
    to ensure they are within valid ranges and compatible with each other.
    Invalid configurations can lead to analysis failures or incorrect results.
    
    Args:
        config: Configuration object to validate containing all analysis parameters
        
    Returns:
        List of validation error messages. Empty list indicates valid configuration.
        Each error message describes a specific validation failure with suggested
        corrections where applicable.
        
    Examples:
        >>> config = get_default_config()
        >>> errors = validate_config(config)
        >>> if errors:
        ...     for error in errors:
        ...         print(f"Configuration error: {error}")
        >>> else:
        ...     print("Configuration is valid")
        
        >>> # Check specific component
        >>> config.thickness.saturation_threshold = 300  # Invalid
        >>> errors = validate_config(config)
        >>> print(errors[0])
        saturation_threshold must be between 1 and 255
    """
    errors = []
    logger = logging.getLogger(__name__)
    
    errors = []
    logger = logging.getLogger(__name__)
    
    # Log validation start
    logger.debug("Starting comprehensive configuration validation")
    
    # Validate that config object exists and is correct type
    if config is None:
        errors.append("Configuration object cannot be None")
        return errors
        
    if not isinstance(config, Config):
        errors.append(f"Expected Config object, got {type(config).__name__}")
        return errors

    # Validate ProcessingConfig
    logger.debug("Validating processing configuration")
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

    # Validate background correction parameters
    if config.processing.basic_correction_n_components < 1:
        errors.append("basic_correction_n_components must be positive")
    if config.processing.rolling_ball_radius < 1:
        errors.append("rolling_ball_radius must be positive")
    if not (0 < config.processing.restore_percentile < 100):
        errors.append("restore_percentile must be between 0 and 100")
    if config.processing.homomorphic_cutoff <= 0:
        errors.append("homomorphic_cutoff must be positive")
    if config.processing.homomorphic_g_low <= 0:
        errors.append("homomorphic_g_low must be positive")
    if config.processing.homomorphic_g_high <= 0:
        errors.append("homomorphic_g_high must be positive")

    # Validate ThicknessConfig
    logger.debug("Validating thickness estimation configuration")
    if (
        config.thickness.saturation_threshold <= 0
        or config.thickness.saturation_threshold > 255
    ):
        errors.append("saturation_threshold must be between 1 and 255")
        
    if config.thickness.attenuation_coefficient <= 0:
        errors.append("attenuation_coefficient must be positive")
        
    if config.thickness.min_transmittance <= 0 or config.thickness.min_transmittance >= 1:
        errors.append("min_transmittance must be between 0 and 1 (exclusive)")
        
    # Validate thickness range
    if len(config.thickness.thickness_range_um) != 2:
        errors.append("thickness_range_um must contain exactly two values")
    elif config.thickness.thickness_range_um[0] >= config.thickness.thickness_range_um[1]:
        errors.append("thickness_range_um[0] must be less than thickness_range_um[1]")
    elif config.thickness.thickness_range_um[0] < 0:
        errors.append("thickness_range_um values must be non-negative")

    # Validate UniformityConfig
    logger.debug("Validating uniformity analysis configuration")
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

    # Validate new ProcessingConfig fields
    if config.processing.basic_correction_n_components < 1:
        errors.append("basic_correction_n_components must be positive")
    if config.processing.rolling_ball_radius < 1:
        errors.append("rolling_ball_radius must be positive")
    if not (0 < config.processing.restore_percentile < 100):
        errors.append("restore_percentile must be between 0 and 100")
    if config.processing.homomorphic_cutoff <= 0:
        errors.append("homomorphic_cutoff must be positive")

    # Validate new ThicknessConfig fields
    if config.thickness.attenuation_coefficient <= 0:
        errors.append("attenuation_coefficient must be positive")

    # Validate GLCM and LBP parameters
    if not all(d > 0 for d in config.uniformity.glcm_distances):
        errors.append("All glcm_distances must be positive")
    if config.uniformity.lbp_radius < 1:
        errors.append("lbp_radius must be positive")
    if config.uniformity.lbp_points < 1:
        errors.append("lbp_points must be positive")

    # Validate RulerDetectionConfig
    logger.debug("Validating ruler detection configuration")
    if config.ruler_detection.min_line_length <= 0:
        errors.append("min_line_length must be positive")
    if config.ruler_detection.max_line_gap < 0:
        errors.append("max_line_gap must be non-negative")
    if config.ruler_detection.expected_tick_distance_mm <= 0:
        errors.append("expected_tick_distance_mm must be positive")
    if config.ruler_detection.canny_threshold1 <= 0:
        errors.append("canny_threshold1 must be positive")
    if config.ruler_detection.canny_threshold2 <= 0:
        errors.append("canny_threshold2 must be positive")
    if config.ruler_detection.canny_threshold1 >= config.ruler_detection.canny_threshold2:
        errors.append("canny_threshold1 must be less than canny_threshold2")
    if config.ruler_detection.hough_threshold <= 0:
        errors.append("hough_threshold must be positive")

    # Validate ShapeDetectionConfig
    logger.debug("Validating shape detection configuration")
    if config.shape_detection.min_area <= 0:
        errors.append("min_area must be positive")
    if config.shape_detection.max_area <= config.shape_detection.min_area:
        errors.append("max_area must be greater than min_area")
    if not (0 < config.shape_detection.approx_epsilon_ratio < 1):
        errors.append("approx_epsilon_ratio must be between 0 and 1")
    if config.shape_detection.min_vertices < 3:
        errors.append("min_vertices must be at least 3")
    if config.shape_detection.max_vertices <= config.shape_detection.min_vertices:
        errors.append("max_vertices must be greater than min_vertices")

    # Validate VisualizationConfig percentile ranges
    if len(config.visualization.heatmap_percentile_range) != 2:
        errors.append("heatmap_percentile_range must contain exactly two values")
    elif (config.visualization.heatmap_percentile_range[0] >= 
          config.visualization.heatmap_percentile_range[1]):
        errors.append("heatmap_percentile_range[0] must be less than heatmap_percentile_range[1]")
    elif (config.visualization.heatmap_percentile_range[0] < 0 or 
          config.visualization.heatmap_percentile_range[1] > 100):
        errors.append("heatmap_percentile_range values must be between 0 and 100")

    # Validate output directory
    if config.output_dir is not None:
        try:
            output_path = Path(config.output_dir)
            # Try to create directory if it doesn't exist (validation only)
            if not output_path.exists():
                logger.debug(f"Output directory {output_path} will be created when needed")
        except Exception as e:
            errors.append(f"Invalid output_dir path: {e}")

    # Log validation completion
    if errors:
        logger.warning(f"Configuration validation found {len(errors)} errors")
        for i, error in enumerate(errors, 1):
            logger.warning(f"  {i}. {error}")
    else:
        logger.debug("Configuration validation completed successfully")

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
