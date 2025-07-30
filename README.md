# ESNF Mat Analyzer

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://img.shields.io/badge/DOI-pending-orange.svg)](https://doi.org/pending)

**Advanced Analysis Tool for Electrospun Nanofiber Mat Uniformity and Structure**

## Summary

The ESNF Mat Analyzer is a comprehensive Python software package designed for quantitative analysis of electrospun nanofiber mats. It provides advanced image processing capabilities, multiple uniformity metrics, and sophisticated visualization tools specifically tailored for materials science research involving nanofiber characterization.

## Statement of Need

Electrospun nanofiber mats are crucial materials in applications ranging from filtration and tissue engineering to energy storage. The uniformity and structural properties of these mats directly impact their performance, yet existing analysis tools often lack the specialized metrics and robust image processing capabilities required for comprehensive characterization. 

Current solutions typically focus on individual fiber properties rather than mat-scale uniformity, provide limited background correction methods, or lack physics-based thickness estimation models. ESNF Mat Analyzer addresses these limitations by providing:

- **Mat-scale uniformity analysis** using advanced FFT-based anisotropy detection
- **Multiple thickness estimation models** including physics-based Beer-Lambert approaches
- **Comprehensive texture analysis** using Gray-Level Co-occurrence Matrix (GLCM) metrics
- **Robust background correction** with multiple algorithms (BASIC, rolling ball, RESTORE, homomorphic)
- **Power spectral density analysis** for frequency domain characterization
- **User-friendly GUI** with real-time visualization and method comparison tools

## Key Features

### Advanced Image Processing
- **Multiple background correction methods**: Default normalization, BASIC, rolling ball, RESTORE, and homomorphic filtering
- **HDR processing support** for high dynamic range imaging
- **Robust preprocessing** with adaptive thresholding and noise reduction
- **Region of Interest (ROI) selection** with interactive tools

### Comprehensive Uniformity Metrics
- **Traditional metrics**: Coefficient of variation, thickness range ratio, Gini coefficient
- **Mat-scale analysis**: Overall uniformity score combining multiple factors
- **Anisotropy detection**: FFT-based directional analysis
- **Texture characterization**: GLCM-based homogeneity, energy, correlation, and contrast
- **Frequency analysis**: Power spectral density uniformity metrics

### Physics-Based Analysis
- **Beer-Lambert thickness model**: Physics-based attenuation modeling
- **Multiple thickness models**: Linear, logarithmic, and exponential options
- **Uncertainty quantification**: Statistical analysis of measurement reliability
- **Spatial scale calibration**: Accurate dimensional analysis

### Visualization and Export
- **Interactive heatmaps** with customizable color schemes
- **Comparative visualizations** for method evaluation
- **Comprehensive reporting** with tabbed results display
- **Data export capabilities** in multiple formats
- **Publication-ready figures** with proper scaling and annotations

## Installation

### Requirements
- Python 3.8 or higher
- NumPy >= 1.20.0
- SciPy >= 1.7.0
- OpenCV >= 4.5.0
- scikit-image >= 0.18.0
- Matplotlib >= 3.3.0
- Pillow >= 8.0.0
- tkinter (usually included with Python)

### Quick Start

1. **Clone the repository:**
```bash
git clone https://github.com/DocRuzzy/esnf-mat-analyzer.git
cd esnf-mat-analyzer
```

2. **Run the automated setup:**
```bash
python setup_environment.py
```

3. **Start the application:**
```bash
python run_esnf_analyzer.py
```

### Alternative Installation Methods

**Using pip (when published):**
```bash
pip install esnf-mat-analyzer
```

**For development:**
```bash
git clone https://github.com/DocRuzzy/esnf-mat-analyzer.git
cd esnf-mat-analyzer
pip install -e .
```

## Usage

### Graphical User Interface

The GUI provides an intuitive interface for nanofiber mat analysis:

1. **Load images** using the file selection panel
2. **Select analysis region** by drawing a rectangle on the image
3. **Configure analysis parameters**:
   - Background correction method
   - Thickness estimation model
   - Processing options
4. **Run analysis** and view comprehensive results
5. **Export results** and visualizations

### Command Line Interface

For batch processing and automation:

```bash
# Analyze a single image
esnf-analyzer input_image.png --output results/ --config config.yaml

# Batch process a directory
esnf-analyzer input_directory/ --output results/ --pattern "*.png"

# Use specific background correction
esnf-analyzer image.png --bg-method rolling_ball --thickness-model beer_lambert
```

### Python API

For integration into research workflows:

```python
from esnf_mat_analyzer.core.analyzer import NanoFiberAnalyzer
from esnf_mat_analyzer.config.config_manager import get_default_config

# Setup analyzer
config = get_default_config()
analyzer = NanoFiberAnalyzer(config)

# Analyze image
result = analyzer.process_image("path/to/image.png", roi=(100, 100, 500, 500))

# Access results
print(f"Overall uniformity: {result.metrics['overall_mat_uniformity']:.4f}")
print(f"Anisotropy index: {result.metrics['anisotropy_index']:.4f}")
```

## Method Comparison and Validation

The software includes built-in tools for comparing different analysis methods:

### Background Correction Comparison
- Visual side-by-side comparison of correction methods
- Statistical analysis of correction effectiveness
- Saturation rate calculation and optimization

### Thickness Model Validation
- Performance metrics for different thickness models
- Signal-to-noise ratio analysis
- Dynamic range evaluation
- Coefficient of variation comparison

## Configuration

The software uses YAML configuration files for reproducible analysis:

```yaml
processing:
  background_correction_method: "rolling_ball"
  leveling:
    enabled: true
    method: "polynomial"

thickness:
  model_type: "beer_lambert"
  calibration_factor: 1.0

analysis:
  uniformity_metrics: ["cv", "gini", "range_ratio"]
  texture_analysis: true
  anisotropy_analysis: true
```

## Output and Results

### Metrics Provided
- **Overall Mat Uniformity Score**: Composite metric (0-1 scale)
- **Anisotropy Index**: Directional uniformity measure
- **Texture Metrics**: Homogeneity, energy, correlation, contrast
- **Traditional Metrics**: CV, Gini coefficient, thickness ratios
- **Power Spectral Density**: Frequency domain uniformity

### Export Formats
- CSV files with detailed metrics
- JSON files for programmatic access
- PNG/PDF figures for publications
- YAML configuration files for reproducibility

## Contributing

We welcome contributions from the research community:

1. **Fork the repository** on GitHub
2. **Create a feature branch** (`git checkout -b feature/new-analysis-method`)
3. **Make your changes** with appropriate tests
4. **Submit a pull request** with a clear description

### Development Setup
```bash
git clone https://github.com/DocRuzzy/esnf-mat-analyzer.git
cd esnf-mat-analyzer
python setup_environment.py
pip install -e ".[dev]"
```

## Testing

Run the test suite to ensure everything works correctly:

```bash
# Run all tests
pytest

# Run specific test categories
pytest -m "not slow"  # Skip slow tests
pytest -m "unit"      # Unit tests only
pytest -m "integration"  # Integration tests only
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use ESNF Mat Analyzer in your research, please cite:

```bibtex
@software{esnf_mat_analyzer,
  title = {{ESNF Mat Analyzer: Advanced Analysis Tool for Electrospun Nanofiber Mat Uniformity}},
  author = {ESNF Mat Analyzer Team},
  year = {2025},
  url = {https://github.com/DocRuzzy/esnf-mat-analyzer},
  doi = {pending}
}
```

## Acknowledgments

- Thanks to the materials science research community for requirements and feedback
- OpenCV and scikit-image teams for robust image processing foundations
- Matplotlib team for excellent visualization capabilities

## Support and Documentation

- **GitHub Issues**: [Report bugs or request features](https://github.com/DocRuzzy/esnf-mat-analyzer/issues)
- **Documentation**: See inline code documentation and examples
- **Examples**: Check the `examples/` directory for sample analyses

## Roadmap

Future developments planned:
- Machine learning-based fiber detection
- 3D analysis capabilities for thick mats
- Advanced statistical modeling
- Web-based interface option
- Integration with microscopy software

---

**Keywords**: nanofiber, electrospinning, image analysis, materials science, uniformity metrics, microscopy, texture analysis, Python
  - Radial Uniformity Index
  - Gini Coefficient
  - Thickness Range Ratio
- **Visualization**:
  - Thickness heatmaps
  - Radial profile plots
  - Uniformity metric visualizations
- **Data Export**: Export results in various formats (CSV, JSON, text)

## Installation

### Prerequisites

- Python 3.7 or higher
- OpenCV
- NumPy
- Matplotlib
- Pandas
- PyYAML

### Install from Source

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/nanofiber-analyzer.git
   cd nanofiber-analyzer
   ```

2. Install the package:
   ```bash
   pip install -e .
   ```

## Usage

### Command-Line Interface

Process a single image:

```bash
python -m nanofiber_analyzer.main --input path/to/image.jpg --output ./results
```

Process multiple images:

```bash
python -m nanofiber_analyzer.main --input path/to/image/directory --pattern "*.jpg" --output ./results
```

Generate a default configuration file:

```bash
python -m nanofiber_analyzer.main --generate-config config.yaml
```

### Python API

#### GUI

To run the graphical user interface:

```bash
python run_gui.py
```

#### Command-Line

```python
from pathlib import Path
from nanofiber_analyzer.main import create_config, setup_dependencies

# Create or load configuration
config = create_config()  # Default configuration
# or load from file:
# from nanofiber_analyzer.main import load_config
# config = create_config(load_config(Path('config.yaml')))

# Setup dependencies and create analyzer
analyzer = setup_dependencies(config)

# Process a single image
result = analyzer.process_image(Path('path/to/image.jpg'))

# Process multiple images
results = analyzer.batch_process(Path('path/to/images'), pattern="*.jpg")
```

## Configuration

The analyzer can be configured through a YAML file or programmatically. The configuration covers:

- **Image Processing**: Blur, contrast, grayscale conversion
- **Circle Detection**: Detection method, parameters
- **Thickness Estimation**: Model type, parameters
- **Uniformity Analysis**: Parameters for uniformity metrics
- **Visualization**: Colormap, size, resolution

Example configuration:

```yaml
# Image processing
processing:
  blur_kernel_size: 5
  contrast_alpha: 1.5
  contrast_beta: 0
  grayscale_conversion: weighted

# Circle detection
circle_detection:
  min_radius: 50
  max_radius: 500
  detection_method: hough
  param1: 50
  param2: 30

# Thickness estimation
thickness:
  model_type: linear
  a: 1.0
  b: 0.0
  saturation_threshold: 250

# Output directory
output_dir: ./output
```

## Scientific Background

### Electrospun Nanofiber (ESNF) Mat Thickness Uniformity

Electrospinning is a widely used technique for fabricating nanofiber mats with applications in filtration, tissue engineering, drug delivery, and more. The uniformity of thickness in these mats is crucial for their performance in many applications.

### Thickness Estimation Model

The software uses the following models to convert brightness to estimated thickness:

1. **Linear Model**: `thickness = a * brightness + b`
2. **Logarithmic Model**: `thickness = a * log(1 + brightness) + b`
3. **Exponential Model**: `thickness = a * (exp(brightness / 255) - 1) + b`

Where:
- `a` is a scaling factor
- `b` is an offset

The logarithmic and exponential models can better handle the saturation effects in images.

### Uniformity Metrics

#### Radial Uniformity Index (RUI)

Measures variation along radial lines from the center. Values closer to 1.0 indicate better uniformity.

#### Gini Coefficient

Measures statistical dispersion of thickness values. Values closer to 0.0 indicate better uniformity (more equal distribution).

#### Thickness Range Ratio (TRR)

Ratio of minimum to maximum thickness. Values closer to 1.0 indicate better uniformity.

## Limitations

- The absolute thickness cannot be determined without calibration
- Brightness saturation can affect the accuracy of thickness estimation
- The analysis assumes that brightness correlates linearly or non-linearly with thickness

## Examples

See the `examples` directory for Jupyter notebooks demonstrating the usage of the package.

## Scale Calibration (Experimental)

This application includes an experimental feature to automatically calibrate the image scale (e.g., pixels per millimeter) if a ruler is present in the photograph. This allows for measurements made on the image to be converted to real-world units.

**Usage:**

1.  Ensure a ruler with clear markings (e.g., millimeter or centimeter ticks) is placed flat and visible within the image, ideally close to the region of interest but not obscuring it.
2.  The program will attempt to detect this ruler and calculate the scale.
3.  The calculated scale is then used to potentially convert thickness estimations or other spatial measurements into metric units (e.g., nm, mm), depending on the specific configuration of the thickness estimation model.

**Configuration:**

Parameters for ruler detection can be adjusted in the configuration file under the `ruler_detection` section. These include settings for line detection sensitivity, expected tick distances, etc. Please refer to the default configuration file generated by the application for more details on these parameters.

**Note:** This feature is experimental. The accuracy of the scale calibration depends on the clarity of the ruler in the image, its orientation, and the detection parameters.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this software in your research, please cite:

```
[Your paper citation information will go here]
```
