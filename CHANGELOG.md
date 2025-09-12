# Changelog

All notable changes to the ESNF Mat Analyzer project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Comprehensive setup script with JOSS compliance features
- Automated environment configuration for development
- Enhanced README with detailed installation and usage instructions
- JOSS paper template and citation files
- Platform-specific convenience scripts (batch/shell)
- Git hooks for automatic environment setup on branch switching

## [1.0.0] - 2025-07-30

### Added
- Initial release of ESNF Mat Analyzer
- Advanced background correction methods (BASIC, rolling ball, RESTORE, homomorphic)
- Multiple thickness estimation models including Beer-Lambert physics-based approach
- Comprehensive uniformity metrics (traditional and mat-scale)
- FFT-based anisotropy analysis
- GLCM texture analysis (homogeneity, energy, correlation, contrast)
- Power spectral density uniformity metrics
- Interactive GUI with real-time visualization
- Method comparison tools for background correction and thickness models
- ROI selection with zoom/pan capabilities
- Configurable analysis parameters via YAML files
- Data export in multiple formats
- Batch processing capabilities
- Command-line interface for automation
- Python API for research workflow integration

### Features
- **Image Processing**:
  - Multiple background correction algorithms
  - HDR processing support
  - Robust preprocessing with adaptive thresholding
  - Interactive ROI selection tools

- **Analysis Capabilities**:
  - Overall mat uniformity scoring (0-1 scale)
  - Anisotropy index calculation
  - Texture characterization using GLCM
  - Traditional uniformity metrics (CV, Gini, range ratio)
  - Frequency domain analysis via PSD

- **Visualization**:
  - Interactive thickness heatmaps
  - Method comparison visualizations
  - Publication-ready figure export
  - Tabbed results display

- **User Interface**:
  - Intuitive tkinter-based GUI
  - Real-time parameter adjustment
  - Method preview and comparison tools
  - Zoom, pan, and scale bar functionality

- **Configuration and Export**:
  - YAML-based configuration management
  - CSV, JSON, and image export options
  - Batch processing with summary reports
  - Reproducible analysis workflows

### Technical Details
- Python 3.8+ compatibility
- Modular architecture with clear separation of concerns
- Comprehensive error handling and validation
- Extensive documentation and examples
- Unit and integration test coverage
- Cross-platform support (Windows, macOS, Linux)

### Dependencies
- NumPy >= 1.20.0
- SciPy >= 1.7.0
- OpenCV >= 4.5.0
- scikit-image >= 0.18.0
- Matplotlib >= 3.3.0
- Pillow >= 8.0.0
- PyYAML >= 5.4.0

## [0.9.0] - 2025-07-15

### Added
- Beta release with core functionality
- Basic GUI implementation
- Initial background correction methods
- Traditional uniformity metrics

### Fixed
- Image loading and display issues
- ROI selection accuracy
- Configuration file parsing

## [0.1.0] - 2025-06-01

### Added
- Initial project structure
- Basic image processing capabilities
- Proof-of-concept analysis pipeline

---

## Version History Notes

### Semantic Versioning Guidelines
- **Major version (X.y.z)**: Incompatible API changes
- **Minor version (x.Y.z)**: New functionality in a backwards compatible manner
- **Patch version (x.y.Z)**: Backwards compatible bug fixes

### Release Process
1. Update version in `pyproject.toml`
2. Update this CHANGELOG
3. Create git tag with version number
4. Build and publish to PyPI (when applicable)
5. Create GitHub release with release notes

### Future Releases
- **v1.1.0**: Machine learning-based fiber detection
- **v1.2.0**: 3D analysis capabilities
- **v2.0.0**: Web-based interface and API restructure
