"""
Professional logging configuration for ESNF Mat Analyzer.

This module provides comprehensive logging configuration with features including:
- Multiple output formats (console, file, structured JSON)
- Configurable log levels for different components
- Rotation and archival of log files
- Performance-aware logging with minimal overhead
- Integration with analysis workflows

Author: ESNF Mat Analyzer Team
License: GNU General Public License v3.0 or later (GPLv3)
"""

import logging
import logging.handlers
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional, Union
from datetime import datetime


class PerformanceFormatter(logging.Formatter):
    """Custom formatter that includes performance timing information."""
    
    def __init__(self, include_timing: bool = True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.include_timing = include_timing
        self.start_time = time.time()
    
    def format(self, record: logging.LogRecord) -> str:
        # Add timing information
        if self.include_timing:
            record.elapsed = time.time() - self.start_time
            record.timestamp = datetime.now().isoformat()
        
        # Add memory usage if available
        try:
            import psutil
            process = psutil.Process()
            record.memory_mb = process.memory_info().rss / 1024 / 1024
        except ImportError:
            record.memory_mb = 0
        
        return super().format(record)


class JSONFormatter(logging.Formatter):
    """Formatter that outputs structured JSON logs."""
    
    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
        }
        
        # Add exception information if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 
                          'pathname', 'filename', 'module', 'lineno', 
                          'funcName', 'created', 'msecs', 'relativeCreated',
                          'thread', 'threadName', 'processName', 'process',
                          'getMessage', 'exc_info', 'exc_text', 'stack_info']:
                log_entry[key] = value
        
        return json.dumps(log_entry, ensure_ascii=False)


class AnalysisLogger:
    """
    Professional logging manager for analysis workflows.
    
    Provides centralized logging configuration with multiple output formats,
    automatic log rotation, and performance monitoring capabilities.
    """
    
    def __init__(
        self,
        name: str = "esnf_mat_analyzer",
        log_dir: Optional[Union[str, Path]] = None,
        level: Union[str, int] = logging.INFO,
        console_output: bool = True,
        file_output: bool = True,
        json_output: bool = False,
        max_file_size: int = 10 * 1024 * 1024,  # 10 MB
        backup_count: int = 5,
        performance_logging: bool = False
    ):
        """
        Initialize the analysis logger.
        
        Args:
            name: Logger name (typically the application name)
            log_dir: Directory for log files (None for temp directory)
            level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            console_output: Enable console logging
            file_output: Enable file logging
            json_output: Enable structured JSON logging
            max_file_size: Maximum size per log file before rotation
            backup_count: Number of backup files to keep
            performance_logging: Enable performance timing in logs
        """
        self.name = name
        self.log_dir = Path(log_dir) if log_dir else Path.cwd() / "logs"
        self.level = level if isinstance(level, int) else getattr(logging, level.upper())
        self.console_output = console_output
        self.file_output = file_output
        self.json_output = json_output
        self.max_file_size = max_file_size
        self.backup_count = backup_count
        self.performance_logging = performance_logging
        
        # Create log directory
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure logging
        self._setup_logging()
    
    def _setup_logging(self) -> None:
        """Configure logging handlers and formatters."""
        # Get root logger
        logger = logging.getLogger(self.name)
        logger.setLevel(self.level)
        
        # Clear existing handlers
        logger.handlers.clear()
        
        # Console handler
        if self.console_output:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(self.level)
            
            # Use performance formatter if enabled
            if self.performance_logging:
                console_format = (
                    '%(asctime)s - %(name)s - %(levelname)s - '
                    '[%(elapsed).2fs] [%(memory_mb).1fMB] - %(message)s'
                )
                formatter = PerformanceFormatter(
                    include_timing=True,
                    fmt=console_format,
                    datefmt='%H:%M:%S'
                )
            else:
                console_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                formatter = logging.Formatter(console_format, datefmt='%H:%M:%S')
            
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        
        # File handler with rotation
        if self.file_output:
            log_file = self.log_dir / f"{self.name}.log"
            file_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=self.max_file_size,
                backupCount=self.backup_count,
                encoding='utf-8'
            )
            file_handler.setLevel(self.level)
            
            file_format = (
                '%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(funcName)s:%(lineno)d - %(message)s'
            )
            file_formatter = logging.Formatter(file_format, datefmt='%Y-%m-%d %H:%M:%S')
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
        
        # JSON handler for structured logging
        if self.json_output:
            json_file = self.log_dir / f"{self.name}.json"
            json_handler = logging.handlers.RotatingFileHandler(
                json_file,
                maxBytes=self.max_file_size,
                backupCount=self.backup_count,
                encoding='utf-8'
            )
            json_handler.setLevel(self.level)
            json_handler.setFormatter(JSONFormatter())
            logger.addHandler(json_handler)
    
    def get_logger(self, name: Optional[str] = None) -> logging.Logger:
        """
        Get a logger instance.
        
        Args:
            name: Logger name (uses default if None)
            
        Returns:
            Configured logger instance
        """
        if name:
            return logging.getLogger(f"{self.name}.{name}")
        return logging.getLogger(self.name)
    
    def log_analysis_start(self, image_path: Union[str, Path], config: Dict[str, Any]) -> None:
        """Log the start of an analysis with configuration details."""
        logger = self.get_logger("analysis")
        logger.info(f"Starting analysis: {Path(image_path).name}")
        logger.debug(f"Configuration: {json.dumps(config, indent=2, default=str)}")
    
    def log_analysis_complete(
        self, 
        image_path: Union[str, Path], 
        metrics: Dict[str, float],
        processing_time: float
    ) -> None:
        """Log the completion of an analysis with results."""
        logger = self.get_logger("analysis")
        logger.info(
            f"Analysis complete: {Path(image_path).name} "
            f"({processing_time:.2f}s) - {len(metrics)} metrics computed"
        )
        
        # Log key metrics at debug level
        for metric_name, value in metrics.items():
            logger.debug(f"Metric {metric_name}: {value:.6f}")
    
    def log_batch_summary(self, results: list, total_time: float) -> None:
        """Log batch processing summary."""
        logger = self.get_logger("batch")
        
        success_count = sum(1 for r in results if r.get("status") == "success")
        error_count = len(results) - success_count
        
        logger.info(
            f"Batch processing complete: {success_count} successful, "
            f"{error_count} failed, {total_time:.2f}s total"
        )
        
        if error_count > 0:
            logger.warning(f"Failed images: {error_count}/{len(results)}")
    
    def log_performance_metrics(self, metrics: Dict[str, Any]) -> None:
        """Log performance metrics."""
        logger = self.get_logger("performance")
        for metric, value in metrics.items():
            logger.info(f"Performance metric {metric}: {value}")
    
    def configure_component_logging(self, component_levels: Dict[str, Union[str, int]]) -> None:
        """
        Configure logging levels for specific components.
        
        Args:
            component_levels: Dictionary mapping component names to log levels
            
        Example:
            >>> logger_manager.configure_component_logging({
            ...     'image_processor': 'DEBUG',
            ...     'thickness_estimator': 'WARNING',
            ...     'uniformity_metrics': 'INFO'
            ... })
        """
        for component, level in component_levels.items():
            level_int = level if isinstance(level, int) else getattr(logging, level.upper())
            component_logger = logging.getLogger(f"{self.name}.{component}")
            component_logger.setLevel(level_int)


def setup_application_logging(
    config: Optional[Dict[str, Any]] = None,
    log_dir: Optional[Union[str, Path]] = None,
    verbose: bool = False,
    quiet: bool = False
) -> AnalysisLogger:
    """
    Setup application-wide logging configuration.
    
    Args:
        config: Logging configuration dictionary
        log_dir: Directory for log files
        verbose: Enable verbose (debug) logging
        quiet: Enable quiet (error-only) logging
        
    Returns:
        Configured AnalysisLogger instance
        
    Examples:
        >>> # Basic setup
        >>> logger_manager = setup_application_logging()
        >>> logger = logger_manager.get_logger("component")
        
        >>> # Verbose logging for debugging
        >>> logger_manager = setup_application_logging(verbose=True)
        
        >>> # Custom configuration
        >>> config = {
        ...     'level': 'DEBUG',
        ...     'file_output': True,
        ...     'json_output': True,
        ...     'performance_logging': True
        ... }
        >>> logger_manager = setup_application_logging(config=config)
    """
    # Default configuration
    default_config = {
        'name': 'esnf_mat_analyzer',
        'level': logging.DEBUG if verbose else (logging.ERROR if quiet else logging.INFO),
        'console_output': not quiet,
        'file_output': True,
        'json_output': False,
        'performance_logging': verbose,
        'max_file_size': 10 * 1024 * 1024,  # 10 MB
        'backup_count': 5
    }
    
    # Override with provided config
    if config:
        default_config.update(config)
    
    # Set log directory
    if log_dir:
        default_config['log_dir'] = log_dir
    
    # Create and return logger manager
    return AnalysisLogger(**default_config)


def get_analysis_logger(name: str = "analysis") -> logging.Logger:
    """
    Get a logger for analysis components.
    
    Args:
        name: Component name
        
    Returns:
        Logger instance
    """
    return logging.getLogger(f"esnf_mat_analyzer.{name}")


# Configure some useful loggers for common components
def configure_default_loggers(verbose: bool = False) -> None:
    """Configure default loggers for common components."""
    level = logging.DEBUG if verbose else logging.INFO
    
    # Suppress verbose third-party library logging unless in debug mode
    if not verbose:
        logging.getLogger('matplotlib').setLevel(logging.WARNING)
        logging.getLogger('PIL').setLevel(logging.WARNING)
        logging.getLogger('opencv').setLevel(logging.WARNING)
        logging.getLogger('skimage').setLevel(logging.WARNING)
        
    # Set appropriate levels for our components
    component_levels = {
        'image_processor': level,
        'thickness_estimator': level,
        'uniformity_metrics': level,
        'shape_detector': level,
        'ruler_detector': level,
        'visualizer': level,
        'data_exporter': level,
    }
    
    for component, comp_level in component_levels.items():
        logger = logging.getLogger(f"esnf_mat_analyzer.{component}")
        logger.setLevel(comp_level)
