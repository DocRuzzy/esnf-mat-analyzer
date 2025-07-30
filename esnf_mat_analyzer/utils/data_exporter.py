"""
Data exporter component for nanofiber thickness analysis.

This module provides functionality for exporting analysis results to various file formats.
"""

import numpy as np
import json
import csv
from pathlib import Path
from typing import Dict, Tuple, List
import logging
import piexif

from esnf_mat_analyzer.core.interfaces import DataExporterInterface


class DataExporter(DataExporterInterface):
    """
    Component for exporting nanofiber analysis data to various file formats.
    """
    
    def __init__(self, config=None):
        """Initialize the data exporter.
        
        Args:
            config: Export configuration (optional)
        """
        self.logger = logging.getLogger(__name__)
        self.config = config
    
    def export_thickness_map(self, thickness_map: np.ndarray, path: Path) -> None:
        """
        Export the thickness map to a CSV file.
        
        Args:
            thickness_map: 2D thickness map
            path: Output file path
        """
        self.logger.debug(f"Exporting thickness map to {path}")
        
        # Ensure parent directory exists
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save as CSV with row and column indices
        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Write header row with column indices
            writer.writerow(['y/x'] + list(range(thickness_map.shape[1])))
            
            # Write data rows with row indices
            for i, row in enumerate(thickness_map):
                writer.writerow([i] + list(row))
    
    def export_metrics(self, metrics: Dict[str, float], path: Path) -> None:
        """
        Export uniformity metrics to a JSON file.
        
        Args:
            metrics: Dictionary of metric names and values
            path: Output file path
        """
        self.logger.debug(f"Exporting metrics to {path}")
        
        # Ensure parent directory exists
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save as JSON
        with open(path, 'w') as f:
            json.dump(metrics, f, indent=4)
    
    def export_summary(self, image_path: Path, center: Tuple[int, int], radius: int,
                      metrics: Dict[str, float], path: Path) -> None:
        """
        Export a summary of the analysis results to a text file.
        
        Args:
            image_path: Path to the original image
            center: Center coordinates (x, y) of the circular region
            radius: Radius of the circular region
            metrics: Dictionary of metric names and values
            path: Output file path
        """
        self.logger.debug(f"Exporting summary to {path}")
        
        # Ensure parent directory exists
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create summary text
        summary = [
            "Nanofiber Thickness Uniformity Analysis Summary",
            "=" * 50,
            f"Image: {image_path.name}",
            f"Date: {Path.ctime(path)}",
            "-" * 50,
            "Region of Interest:",
            f"  Center: ({center[0]}, {center[1]})",
            f"  Radius: {radius} pixels",
            f"  Area: {np.pi * radius * radius:.1f} square pixels",
            "-" * 50,
            "Uniformity Metrics:"
        ]
        
        # Add metrics
        for name, value in metrics.items():
            summary.append(f"  {name}: {value:.4f}")
        
        # Add metric descriptions
        summary.extend([
            "-" * 50,
            "Metric Descriptions:",
            "  Radial Uniformity Index: Measures thickness variation along radial lines.",
            "    - Higher values (closer to 1.0) indicate better uniformity.",
            "  Gini Coefficient: Measures statistical dispersion of thickness values.",
            "    - Lower values (closer to 0.0) indicate better uniformity.",
            "  Thickness Range Ratio: Ratio of minimum to maximum thickness.",
            "    - Higher values (closer to 1.0) indicate better uniformity.",
            "-" * 50,
            "Notes:",
            "  - Thickness values are relative and not calibrated to absolute units.",
            "  - Saturated regions may affect the accuracy of uniformity metrics."
        ])
        
        # Write to file
        with open(path, 'w') as f:
            f.write('\n'.join(summary))
    
    def export_radial_data(self, radial_profiles: List[np.ndarray], path: Path) -> None:
        """
        Export radial profile data to a CSV file.
        
        Args:
            radial_profiles: List of radial profiles
            path: Output file path
        """
        self.logger.debug(f"Exporting radial data to {path}")
        
        # Ensure parent directory exists
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Find the maximum length of any profile
        max_length = max(len(profile) for profile in radial_profiles)
        
        # Create a 2D array with profiles as columns
        data = np.zeros((max_length, len(radial_profiles)))
        
        # Fill the array with profile values
        for i, profile in enumerate(radial_profiles):
            data[:len(profile), i] = profile
        
        # Save as CSV
        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Write header row
            angles = np.linspace(0, 360, len(radial_profiles), endpoint=False)
            header = ['Distance'] + [f"{angle:.1f}°" for angle in angles]
            writer.writerow(header)
            
            # Write data rows
            for i in range(max_length):
                row = [i] + list(data[i, :])
                writer.writerow(row)

    def write_scale_to_metadata(self, image_path: Path, scale: float) -> None:
        """
        Write the scale to the image's EXIF metadata.

        Args:
            image_path: Path to the image file.
            scale: The scale in pixels per millimeter.
        """
        try:
            exif_dict = piexif.load(str(image_path))
        except Exception:
            exif_dict = {"0th": {}, "Exif": {}, "GPS": {}, "1st": {}, "thumbnail": None}

        user_comment = f"pixels_per_mm:{scale}"
        exif_dict["Exif"][piexif.ExifIFD.UserComment] = piexif.helper.UserComment.dump(user_comment, encoding="unicode")

        try:
            exif_bytes = piexif.dump(exif_dict)
            piexif.insert(exif_bytes, str(image_path))
            self.logger.info(f"Successfully wrote scale to {image_path}")
        except Exception as e:
            self.logger.error(f"Failed to write EXIF data to {image_path}: {e}")
