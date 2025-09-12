"""
Background Correction Module for ESNF Mat Analyzer

This module provides multiple approaches to background correction for electrospun
nanofiber mat analysis, each optimized for different scenarios and dependencies.

Available Approaches:
--------------------

1. **Advanced (advanced.py)**
   - Primary implementation with comprehensive 4-step scientific workflow
   - Implements the complete ESNF background correction guide
   - Supports multiple correction methods (BASIC, rolling ball, RESTORE, homomorphic)
   - Used by the main GUI and analysis pipeline

2. **Mat-Level (mat_level.py)**
   - Specialized for mat-level analysis where individual fibers are not visible
   - Focuses on preserving continuous mat regions and thickness gradients
   - Optimized for thick mats and continuous regions
   - Uses adaptive polynomial surface fitting

3. **Enhanced Scipy (enhanced_scipy.py)**
   - Advanced implementation using scipy for optimization and signal processing
   - Includes true RESTORE method implementation with iterative optimization
   - ROI perimeter-based background estimation with polynomial gradient modeling
   - Requires scipy dependencies

4. **Enhanced OpenCV (enhanced_opencv.py)**
   - OpenCV-only implementation for environments with minimal dependencies
   - Fast bilateral filtering and Gaussian-based illumination correction
   - ROI-aware processing with robust statistical background estimation
   - No scipy dependencies required

Usage Examples:
--------------

```python
# Primary approach (used by GUI)
from esnf_mat_analyzer.processing.background_correction.advanced import AdvancedBackgroundProcessor
processor = AdvancedBackgroundProcessor()
corrected = processor.comprehensive_correction(image, method="restore")

# Mat-level approach for thick mats
from esnf_mat_analyzer.processing.background_correction.mat_level import MatLevelBackgroundProcessor
processor = MatLevelBackgroundProcessor()
corrected = processor.adaptive_polynomial_surface_fitting(image, roi_mask)

# Enhanced approaches for specialized use cases
from esnf_mat_analyzer.processing.background_correction.enhanced_scipy import EnhancedScipyBackgroundProcessor
from esnf_mat_analyzer.processing.background_correction.enhanced_opencv import EnhancedOpenCVBackgroundProcessor
```

Selection Guidelines:
--------------------

- **Use Advanced** for standard analysis and GUI integration
- **Use Mat-Level** for thick mats where individual fibers are not visible
- **Use Enhanced Scipy** for research requiring advanced optimization methods
- **Use Enhanced OpenCV** for deployment environments with minimal dependencies

All approaches are under active development and support ROI-aware processing.
"""

# Import the main classes for easy access
from .advanced import AdvancedBackgroundProcessor
from .mat_level import MatLevelBackgroundProcessor

try:
    from .enhanced_scipy import EnhancedScipyBackgroundProcessor
except ImportError:
    # scipy not available
    EnhancedScipyBackgroundProcessor = None

try:
    from .enhanced_opencv import EnhancedOpenCVBackgroundProcessor
except ImportError:
    # Should not happen as OpenCV is a core dependency
    EnhancedOpenCVBackgroundProcessor = None

__all__ = [
    'AdvancedBackgroundProcessor',
    'MatLevelBackgroundProcessor', 
    'EnhancedScipyBackgroundProcessor',
    'EnhancedOpenCVBackgroundProcessor'
]

# Version info for the background correction module
__version__ = "1.0.0"
__author__ = "ESNF Mat Analyzer Team"
