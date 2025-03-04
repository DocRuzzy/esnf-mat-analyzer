"""
# Nanofiber Thickness Uniformity Analysis Example

This notebook demonstrates how to use the nanofiber_analyzer package for analyzing the thickness uniformity of electrospun nanofiber (ESNF) mats.
"""

# %% [markdown]
# ## Setup and Configuration

# %%
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pandas as pd
import yaml
import json

# Add the package to the path (if it's not installed)
# sys.path.append('/path/to/nanofiber_analyzer')

# Import the necessary components
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
from nanofiber_analyzer.core.analyzer import NanoFiberAnalyzer

# %% [markdown]
# ## Create Configuration Objects

# %%
# Create default configurations
processing_config = ProcessingConfig(
    blur_kernel_size=5,
    contrast_alpha=1.5,
    contrast_beta=0,
    grayscale_conversion='weighted'
)

circle_detection_config = CircleDetectionConfig(
    min_radius=50,
    max_radius=500,
    detection_method='hough',
    param1=50,
    param2=30
)

thickness_config = ThicknessConfig(
    model_type='linear',
    a=1.0,
    b=0.0,
    saturation_threshold=250
)

uniformity_config = UniformityConfig(
    num_radial_lines=36,
    bin_count=50,
    smoothing_factor=0.5
)

visualization_config = VisualizationConfig(
    colormap='viridis',
    dpi=300,
    figure_size=(10, 8),
    show_saturated=True
)

# Create output directory
output_dir = Path('./output')
output_dir.mkdir(parents=True, exist_ok=True)

# Create main configuration
config = Config(
    processing=processing_config,
    circle_detection=circle_detection_config,
    thickness=thickness_config,
    uniformity=uniformity_config,
    visualization=visualization_config,
    output_dir=output_dir
)

# %% [markdown]
# ## Load Configuration from File
# 
# Alternatively, you can load configuration from a YAML file.

# %%
# Uncomment to load from file
# with open('config.yaml', 'r') as f:
#     config_data = yaml.safe_load(f)
# 
# from nanofiber_analyzer.main import create_config
# config = create_config(config_data)

# %% [markdown]
# ## Initialize Components

# %%
# Create image processor
image_processor = ImageProcessor(config.processing)

# Create circle detector
circle_detector = CircleDetector(config.circle_detection)

# Create thickness estimator
thickness_estimator = ThicknessEstimator(config.thickness)

# Create uniformity metrics
uniformity_metrics = [
    RadialUniformityIndex(config.uniformity),
    GiniCoefficient(),
    ThicknessRangeRatio()
]

# Create visualizer
visualizer = Visualizer(config.visualization)

# Create data exporter
data_exporter = DataExporter()

# %% [markdown]
# ## Create Analyzer

# %%
# Create analyzer
analyzer = NanoFiberAnalyzer(
    config=config,
    image_processor=image_processor,
    circle_detector=circle_detector,
    thickness_estimator=thickness_estimator,
    uniformity_metrics=uniformity_metrics,
    visualizer=visualizer,
    data_exporter=data_exporter
)

# %% [markdown]
# ## Process a Single Image

# %%
# Set image path
image_path = Path('path/to/your/nanofiber_image.jpg')

# Process the image
result = analyzer.process_image(image_path)

# %% [markdown]
# ## Examine Results

# %%
# Print basic results
print(f"Image: {result['image_path']}")
print(f"Circle center: {result['center']}")
print(f"Circle radius: {result['radius']}")
print("\nUniformity Metrics:")
for name, value in result['metrics'].items():
    print(f"  {name}: {value:.4f}")

# %% [markdown]
# ## Load and Display Thickness Map

# %%
# Load thickness map from CSV
thickness_map_path = Path(result['results_dir']) / "thickness_map.csv"
thickness_df = pd.read_csv(thickness_map_path, index_col=0)

# Convert to numpy array
thickness_map = thickness_df.values

# Display as heatmap
plt.figure(figsize=(10, 8))
plt.imshow(thickness_map, cmap='viridis')
plt.colorbar(label='Estimated Thickness')
plt.title('Nanofiber Thickness Map')
plt.xlabel('X (pixels)')
plt.ylabel('Y (pixels)')
plt.show()

# %% [markdown]
# ## Create Custom Visualizations

# %%
# Load metrics
metrics_path = Path(result['results_dir']) / "metrics.json"
with open(metrics_path, 'r') as f:
    metrics = json.load(f)

# Create thickness histogram
plt.figure(figsize=(12, 6))

# Filter out zero values (background)
valid_thickness = thickness_map[thickness_map > 0]

# Create histogram
plt.hist(valid_thickness.flatten(), bins=50, alpha=0.7, color='skyblue', edgecolor='black')
plt.title('Thickness Distribution')
plt.xlabel('Thickness')
plt.ylabel('Frequency')
plt.grid(alpha=0.3)
plt.show()

# %% [markdown]
# ## Batch Process Multiple Images

# %%
# Specify directory containing images
image_dir = Path('path/to/your/images')

# Define file pattern (default: *.jpg)
pattern = "*.jpg"

# Process all matching images
batch_results = analyzer.batch_process(image_dir, pattern)

# %% [markdown]
# ## Compare Metrics Across Multiple Samples

# %%
# Extract metrics from batch results
sample_names = []
radial_uniformity = []
gini_coefficients = []
thickness_ratios = []

for result in batch_results:
    # Extract sample name from image path
    sample_name = Path(result['image_path']).stem
    sample_names.append(sample_name)
    
    # Extract metrics
    metrics = result['metrics']
    radial_uniformity.append(metrics.get('Radial Uniformity Index', 0))
    gini_coefficients.append(metrics.get('Gini Coefficient', 0))
    thickness_ratios.append(metrics.get('Thickness Range Ratio', 0))

# Create DataFrame for comparison
comparison_df = pd.DataFrame({
    'Sample': sample_names,
    'Radial Uniformity Index': radial_uniformity,
    'Gini Coefficient': gini_coefficients,
    'Thickness Range Ratio': thickness_ratios
})

# Display comparison table
comparison_df

# %% [markdown]
# ## Create Comparison Plots

# %%
# Set up figure with multiple subplots
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12))

# Plot Radial Uniformity Index (higher is better)
ax1.bar(sample_names, radial_uniformity, color='skyblue')
ax1.set_title('Radial Uniformity Index (higher is better)')
ax1.set_ylim(0, 1)
ax1.grid(axis='y', alpha=0.3)
ax1.set_xticklabels(sample_names, rotation=45, ha='right')

# Plot Gini Coefficient (lower is better)
ax2.bar(sample_names, gini_coefficients, color='lightgreen')
ax2.set_title('Gini Coefficient (lower is better)')
ax2.set_ylim(0, 1)
ax2.grid(axis='y', alpha=0.3)
ax2.set_xticklabels(sample_names, rotation=45, ha='right')

# Plot Thickness Range Ratio (higher is better)
ax3.bar(sample_names, thickness_ratios, color='salmon')
ax3.set_title('Thickness Range Ratio (higher is better)')
ax3.set_ylim(0, 1)
ax3.grid(axis='y', alpha=0.3)
ax3.set_xticklabels(sample_names, rotation=45, ha='right')

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Testing Different Thickness Models
# 
# You can test different models for converting brightness to thickness.

# %%
# Create a copy of the configuration
config_log = Config(
    processing=config.processing,
    circle_detection=config.circle_detection,
    thickness=ThicknessConfig(
        model_type='logarithmic',
        a=1.0,
        b=0.0,
        saturation_threshold=250
    ),
    uniformity=config.uniformity,
    visualization=config.visualization,
    output_dir=output_dir / "logarithmic_model"
)

# Create analyzer with logarithmic model
analyzer_log = NanoFiberAnalyzer(
    config=config_log,
    image_processor=image_processor,
    circle_detector=circle_detector,
    thickness_estimator=ThicknessEstimator(config_log.thickness),
    uniformity_metrics=uniformity_metrics,
    visualizer=visualizer,
    data_exporter=data_exporter
)

# Process the same image with logarithmic model
result_log = analyzer_log.process_image(image_path)

# Compare metrics
print("Linear Model Metrics:")
for name, value in result['metrics'].items():
    print(f"  {name}: {value:.4f}")

print("\nLogarithmic Model Metrics:")
for name, value in result_log['metrics'].items():
    print(f"  {name}: {value:.4f}")

# %% [markdown]
# ## Conclusion
# 
# This notebook demonstrated how to use the nanofiber_analyzer package for analyzing the thickness uniformity of electrospun nanofiber mats. The package provides a modular, extensible framework for image processing, thickness estimation, and uniformity analysis.
