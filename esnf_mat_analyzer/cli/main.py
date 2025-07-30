"""
Professional command-line interface for ESNF Mat Analyzer.

This module provides a comprehensive CLI with features including:
- Professional argument parsing and validation
- Progress bars for batch processing
- Colored output for better user experience
- Configuration validation and management
- Detailed error reporting and logging
- Support for various input/output formats

Author: ESNF Mat Analyzer Team
License: GNU General Public License v3.0 or later (GPLv3)
"""

import argparse
import sys
import json
import time
from pathlib import Path
from typing import List, Optional, Dict, Any, Union
import logging

try:
    # Optional dependency for colored output
    from colorama import init, Fore, Back, Style
    init(autoreset=True)
    HAS_COLORAMA = True
except ImportError:
    HAS_COLORAMA = False
    # Fallback to no-op if colorama not available
    class Fore:
        RED = GREEN = YELLOW = BLUE = MAGENTA = CYAN = WHITE = RESET = ""
    class Back:
        RED = GREEN = YELLOW = BLUE = MAGENTA = CYAN = WHITE = RESET = ""  
    class Style:
        DIM = NORMAL = BRIGHT = RESET_ALL = ""

try:
    # Optional dependency for progress bars
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

from ..core.analyzer import NanoFiberAnalyzer
from ..core.data_types import Config
from ..config.config_manager import (
    load_config, save_config, validate_config, 
    get_default_config, generate_default_config_file
)


class CLIColors:
    """Color constants for CLI output."""
    SUCCESS = Fore.GREEN
    ERROR = Fore.RED
    WARNING = Fore.YELLOW
    INFO = Fore.BLUE
    HIGHLIGHT = Fore.MAGENTA
    DIM = Style.DIM
    BRIGHT = Style.BRIGHT
    RESET = Style.RESET_ALL


def print_colored(message: str, color: str = "", bold: bool = False) -> None:
    """Print colored message to stdout."""
    if bold:
        message = Style.BRIGHT + message
    print(f"{color}{message}{Style.RESET_ALL}")


def print_error(message: str) -> None:
    """Print error message to stderr."""
    print(f"{CLIColors.ERROR}ERROR: {message}{CLIColors.RESET}", file=sys.stderr)


def print_warning(message: str) -> None:
    """Print warning message."""
    print(f"{CLIColors.WARNING}WARNING: {message}{CLIColors.RESET}")


def print_success(message: str) -> None:
    """Print success message."""
    print(f"{CLIColors.SUCCESS}SUCCESS: {message}{CLIColors.RESET}")


def print_info(message: str) -> None:
    """Print info message."""
    print(f"{CLIColors.INFO}INFO: {message}{CLIColors.RESET}")


def setup_logging(verbose: bool = False, quiet: bool = False) -> None:
    """
    Configure logging for the CLI application.
    
    Args:
        verbose: Enable debug-level logging
        quiet: Suppress all but error messages
    """
    if quiet:
        level = logging.ERROR
    elif verbose:
        level = logging.DEBUG
    else:
        level = logging.INFO
    
    # Configure root logger
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Suppress verbose third-party library logging unless in debug mode
    if not verbose:
        logging.getLogger('matplotlib').setLevel(logging.WARNING)
        logging.getLogger('PIL').setLevel(logging.WARNING)


def validate_input_path(input_path: Union[str, Path]) -> Path:
    """
    Validate and convert input path.
    
    Args:
        input_path: Input path as string or Path object
        
    Returns:
        Validated Path object
        
    Raises:
        FileNotFoundError: If path doesn't exist
        ValueError: If path is neither file nor directory
    """
    path = Path(input_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Input path does not exist: {path}")
    
    if not (path.is_file() or path.is_dir()):
        raise ValueError(f"Input path must be a file or directory: {path}")
    
    return path


def validate_config_file(config_path: Union[str, Path]) -> Config:
    """
    Load and validate configuration file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Validated configuration object
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If configuration is invalid
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    print_info(f"Loading configuration from: {config_path}")
    
    # Load configuration
    try:
        config = load_config(config_path)
    except Exception as e:
        raise ValueError(f"Failed to load configuration: {e}")
    
    # Validate configuration
    errors = validate_config(config)
    if errors:
        print_error("Configuration validation failed:")
        for i, error in enumerate(errors, 1):
            print(f"  {i}. {error}")
        raise ValueError(f"Configuration has {len(errors)} validation errors")
    
    print_success("Configuration loaded and validated successfully")
    return config


def create_progress_bar(total: int, desc: str = "Processing") -> Optional[Any]:
    """Create progress bar if tqdm is available."""
    if HAS_TQDM:
        return tqdm(total=total, desc=desc, unit="image")
    return None


def process_single_image(
    analyzer: NanoFiberAnalyzer, 
    image_path: Path,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Process a single image with error handling.
    
    Args:
        analyzer: Configured analyzer instance
        image_path: Path to image file
        verbose: Enable verbose output
        
    Returns:
        Processing result summary
    """
    if verbose:
        print_info(f"Processing: {image_path.name}")
    
    start_time = time.time()
    
    try:
        # Validate image before processing
        if not analyzer.validate_image(image_path):
            raise ValueError(f"Invalid or unsupported image format")
        
        # Process image
        result = analyzer.process_image(image_path)
        
        processing_time = time.time() - start_time
        
        # Create summary
        summary = {
            "image_path": str(image_path),
            "status": "success",
            "processing_time": processing_time,
            "metrics": result.metrics,
            "warnings": result.validate_completeness()
        }
        
        if verbose:
            print_success(f"Completed: {image_path.name} ({processing_time:.2f}s)")
            
            # Show key metrics
            key_metrics = ["overall_mat_uniformity", "anisotropy_index"]
            for metric in key_metrics:
                value = result.get_metric(metric)
                if value is not None:
                    print(f"  {metric}: {value:.4f}")
        
        return summary
        
    except Exception as e:
        processing_time = time.time() - start_time
        error_msg = str(e)
        
        if verbose:
            print_error(f"Failed: {image_path.name} - {error_msg}")
        
        return {
            "image_path": str(image_path),
            "status": "error",
            "processing_time": processing_time,
            "error_message": error_msg
        }


def process_batch(
    analyzer: NanoFiberAnalyzer,
    input_dir: Path,
    pattern: str = "*.tif",
    verbose: bool = False
) -> List[Dict[str, Any]]:
    """
    Process multiple images in batch with progress tracking.
    
    Args:
        analyzer: Configured analyzer instance
        input_dir: Directory containing images
        pattern: File pattern for image selection
        verbose: Enable verbose output
        
    Returns:
        List of processing result summaries
    """
    # Find matching images
    image_files = sorted(list(input_dir.glob(pattern)))
    
    if not image_files:
        print_warning(f"No images found matching pattern '{pattern}' in {input_dir}")
        return []
    
    print_info(f"Found {len(image_files)} images to process")
    
    # Create progress bar
    progress_bar = create_progress_bar(len(image_files), "Processing images")
    
    results = []
    success_count = 0
    
    for image_path in image_files:
        # Process image
        result = process_single_image(analyzer, image_path, verbose)
        results.append(result)
        
        if result["status"] == "success":
            success_count += 1
        
        # Update progress bar
        if progress_bar:
            progress_bar.update(1)
            progress_bar.set_postfix({
                "Success": f"{success_count}/{len(results)}",
                "Current": image_path.name[:20]
            })
    
    # Close progress bar
    if progress_bar:
        progress_bar.close()
    
    # Print summary
    error_count = len(results) - success_count
    print_success(f"Batch processing completed: {success_count} successful, {error_count} failed")
    
    return results


def save_batch_results(
    results: List[Dict[str, Any]], 
    output_path: Path,
    verbose: bool = False
) -> None:
    """
    Save batch processing results to JSON file.
    
    Args:
        results: List of processing results
        output_path: Path to save results
        verbose: Enable verbose output
    """
    try:
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare summary data
        summary = {
            "metadata": {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "total_images": len(results),
                "successful": sum(1 for r in results if r["status"] == "success"),
                "failed": sum(1 for r in results if r["status"] == "error")
            },
            "results": results
        }
        
        # Save to file
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        if verbose:
            print_success(f"Results saved to: {output_path}")
            
    except Exception as e:
        print_error(f"Failed to save results: {e}")


def create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser."""
    parser = argparse.ArgumentParser(
        prog="esnf-analyzer",
        description="Professional electrospun nanofiber mat uniformity analyzer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze single image
  esnf-analyzer -i sample.tif -o results/
  
  # Batch process directory
  esnf-analyzer -i images/ -p "*.tif" -o results/ -v
  
  # Use custom configuration
  esnf-analyzer -i sample.tif -c config.yaml -o results/
  
  # Generate default configuration
  esnf-analyzer --generate-config config.yaml
  
  # Validate configuration
  esnf-analyzer --validate-config config.yaml
        """
    )
    
    # Input/output arguments
    parser.add_argument(
        "-i", "--input",
        type=str,
        help="Input image file or directory containing images"
    )
    
    parser.add_argument(
        "-o", "--output", 
        type=str,
        default="./output",
        help="Output directory for analysis results (default: ./output)"
    )
    
    parser.add_argument(
        "-p", "--pattern",
        type=str, 
        default="*.tif",
        help="File pattern for batch processing (default: *.tif)"
    )
    
    # Configuration arguments
    parser.add_argument(
        "-c", "--config",
        type=str,
        help="Path to configuration file (YAML format)"
    )
    
    parser.add_argument(
        "--generate-config",
        type=str,
        metavar="PATH",
        help="Generate default configuration file at specified path"
    )
    
    parser.add_argument(
        "--validate-config",
        type=str,
        metavar="PATH", 
        help="Validate configuration file and exit"
    )
    
    # Logging and output control
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output and debug logging"
    )
    
    parser.add_argument(
        "-q", "--quiet",
        action="store_true", 
        help="Suppress all output except errors"
    )
    
    # Analysis options
    parser.add_argument(
        "--scale",
        type=float,
        help="Manual spatial scale in pixels per millimeter"
    )
    
    parser.add_argument(
        "--roi",
        type=str,
        help="Region of interest as 'x,y,width,height' (pixel coordinates)"
    )
    
    # Output format options
    parser.add_argument(
        "--format",
        choices=["json", "csv", "yaml"],
        default="json",
        help="Output format for batch results (default: json)"
    )
    
    # Version information
    parser.add_argument(
        "--version",
        action="version",
        version="ESNF Mat Analyzer 1.0.0"
    )
    
    return parser


def parse_roi_string(roi_str: str) -> tuple:
    """
    Parse ROI string into tuple of integers.
    
    Args:
        roi_str: ROI string in format "x,y,width,height"
        
    Returns:
        Tuple of (x, y, width, height)
        
    Raises:
        ValueError: If ROI string is invalid
    """
    try:
        parts = [int(x.strip()) for x in roi_str.split(',')]
        if len(parts) != 4:
            raise ValueError("ROI must have exactly 4 values")
        if any(x < 0 for x in parts):
            raise ValueError("ROI values must be non-negative")
        return tuple(parts)
    except ValueError as e:
        raise ValueError(f"Invalid ROI format '{roi_str}': {e}")


def main_cli() -> int:
    """
    Main CLI entry point with comprehensive error handling.
    
    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Handle configuration generation
    if args.generate_config:
        try:
            config_path = Path(args.generate_config)
            generate_default_config_file(config_path)
            print_success(f"Default configuration generated: {config_path}")
            return 0
        except Exception as e:
            print_error(f"Failed to generate configuration: {e}")
            return 1
    
    # Handle configuration validation
    if args.validate_config:
        try:
            config = validate_config_file(args.validate_config)
            print_success("Configuration is valid")
            
            # Print configuration summary
            info = {
                "Processing method": config.processing.background_correction_method.name,
                "Thickness model": config.thickness.model_type.name,
                "Output directory": str(config.output_dir),
                "Ruler detection": config.ruler_detection.enabled
            }
            
            print("\nConfiguration Summary:")
            for key, value in info.items():
                print(f"  {key}: {value}")
            
            return 0
        except Exception as e:
            print_error(f"Configuration validation failed: {e}")
            return 1
    
    # Validate required arguments
    if not args.input:
        print_error("Input path is required. Use -i/--input to specify input file or directory.")
        print("Use --help for usage information.")
        return 1
    
    # Setup logging
    setup_logging(verbose=args.verbose, quiet=args.quiet)
    
    try:
        # Validate input path
        input_path = validate_input_path(args.input)
        
        # Load configuration
        if args.config:
            config = validate_config_file(args.config)
        else:
            if not args.quiet:
                print_info("Using default configuration")
            config = get_default_config()
        
        # Override output directory
        config.output_dir = Path(args.output)
        
        # Create analyzer
        if not args.quiet:
            print_info("Initializing analyzer...")
        
        analyzer = NanoFiberAnalyzer.from_config(config)
        
        # Parse ROI if provided
        roi = None
        if args.roi:
            roi = parse_roi_string(args.roi)
            if not args.quiet:
                print_info(f"Using ROI: {roi}")
        
        # Process input
        if input_path.is_file():
            # Single file processing
            if not args.quiet:
                print_info(f"Processing single image: {input_path.name}")
            
            result_summary = process_single_image(
                analyzer, input_path, verbose=args.verbose
            )
            
            if result_summary["status"] == "success":
                print_success("Image processing completed successfully")
                
                # Show warnings if any
                warnings = result_summary.get("warnings", [])
                if warnings:
                    print_warning("Analysis warnings:")
                    for warning in warnings:
                        print(f"  - {warning}")
                
                return 0
            else:
                print_error(f"Processing failed: {result_summary['error_message']}")
                return 1
                
        else:
            # Batch processing
            if not args.quiet:
                print_info(f"Starting batch processing: {input_path}")
            
            results = process_batch(
                analyzer, input_path, args.pattern, verbose=args.verbose
            )
            
            if not results:
                print_warning("No images were processed")
                return 1
            
            # Save results
            output_file = config.output_dir / f"batch_results.{args.format}"
            save_batch_results(results, output_file, verbose=args.verbose)
            
            # Check if any processing succeeded
            success_count = sum(1 for r in results if r["status"] == "success")
            if success_count == 0:
                print_error("All images failed to process")
                return 1
            
            return 0
            
    except KeyboardInterrupt:
        print_warning("Processing interrupted by user")
        return 130
    except Exception as e:
        print_error(f"Unexpected error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main_cli())
