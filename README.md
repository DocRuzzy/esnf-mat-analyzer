# Nanofiber Thickness Uniformity Analyzer

A Python package for analyzing the thickness uniformity of electrospun nanofiber (ESNF) mats from color images. The software converts brightness in images to estimated thickness and calculates various uniformity metrics to quantify the distribution of fibers across the mat.

## Features

- **Image Processing**: Load and preprocess images for analysis
- **Automatic ROI Detection**: Detect circular regions of interest
- **Thickness Estimation**: Convert brightness to estimated thickness
- **Uniformity Metrics**:
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
