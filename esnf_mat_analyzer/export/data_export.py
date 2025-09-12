from typing import Dict, List
from pathlib import Path

class PublicationDataExporter:
    """Export analysis data in formats suitable for publication and archival."""

    def export_analysis_data(self, results: Dict,
                           metadata: Dict,
                           output_dir: Path,
                           formats: List[str] = ['csv', 'hdf5', 'json']) -> None:
        """Export complete analysis dataset.

        Args:
            results: Analysis results dictionary
            metadata: Sample and processing metadata
            output_dir: Output directory
            formats: List of export formats
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        if 'csv' in formats:
            self._export_csv(results, metadata, output_dir)

        if 'hdf5' in formats:
            self._export_hdf5(results, metadata, output_dir)

        if 'json' in formats:
            self._export_json(results, metadata, output_dir)

    def _export_csv(self, results: Dict, metadata: Dict, output_dir: Path) -> None:
        """Export tabular data in CSV format."""
        import pandas as pd

        # Uniformity metrics summary
        metrics_data = {
            'Metric': [],
            'Value': [],
            'Unit': [],
            'Uncertainty': [],
            'Description': []
        }

        # Add all uniformity metrics
        uniformity_metrics = {
            'Radial Uniformity Index': (results.get('rui', 0), 'dimensionless', 0.01, 'Higher = more uniform'),
            'Gini Coefficient': (results.get('gini', 0), 'dimensionless', 0.01, 'Lower = more uniform'),
            'Thickness Range Ratio': (results.get('trr', 0), 'dimensionless', 0.01, 'Higher = more uniform'),
            'Mean Thickness': (results.get('mean_thickness', 0), 'μm', 0.1, 'Average thickness'),
            'Thickness Std Dev': (results.get('thickness_std', 0), 'μm', 0.1, 'Thickness variability')
        }

        for metric_name, (value, unit, uncertainty, description) in uniformity_metrics.items():
            metrics_data['Metric'].append(metric_name)
            metrics_data['Value'].append(value)
            metrics_data['Unit'].append(unit)
            metrics_data['Uncertainty'].append(uncertainty)
            metrics_data['Description'].append(description)

        df_metrics = pd.DataFrame(metrics_data)
        df_metrics.to_csv(output_dir / 'uniformity_metrics.csv', index=False)

        # Export thickness map data
        if 'thickness_map' in results:
            thickness_df = pd.DataFrame(results['thickness_map'])
            thickness_df.to_csv(output_dir / 'thickness_map.csv', index=False)

    def _export_hdf5(self, results: Dict, metadata: Dict, output_dir: Path) -> None:
        """Export data in HDF5 format."""
        # This is a placeholder implementation.
        pass

    def _export_json(self, results: Dict, metadata: Dict, output_dir: Path) -> None:
        """Export data in JSON format."""
        import json

        # Combine results and metadata
        export_data = {
            'metadata': metadata,
            'results': results
        }

        # Convert numpy arrays to lists for JSON serialization
        def convert(o):
            if isinstance(o, np.ndarray):
                return o.tolist()
            raise TypeError

        with open(output_dir / 'analysis_data.json', 'w') as f:
            json.dump(export_data, f, default=convert, indent=4)
