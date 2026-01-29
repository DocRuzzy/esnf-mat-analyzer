# ESNF Mat Analyzer - AI Coding Agent Instructions

## Project Overview
Scientific Python package for electrospun nanofiber mat analysis with a focus on **background correction methods**, **uniformity metrics**, and **thickness calibration**. The project follows JOSS (Journal of Open Source Software) standards with a modular architecture built around dependency injection and interface patterns.

## Architecture & Key Components

### Core Pipeline (Process Order)
1. **Image Loading** → `esnf_mat_analyzer/processing/image_processor.py`
2. **Ruler Detection** → `esnf_mat_analyzer/processing/ruler_detector.py` (spatial calibration)
3. **Three-Region Detection** → `esnf_mat_analyzer/processing/region_detector.py` (background/gel/mat segmentation)
4. **Background Correction** → `esnf_mat_analyzer/processing/background_correction/` (4 methods)
5. **ROI Detection** → `esnf_mat_analyzer/processing/shape_detector.py` (gel boundary detection)
6. **Intensity Calibration** → `esnf_mat_analyzer/processing/calibration.py` (0-100 normalized scale)
7. **Thickness Estimation** → `esnf_mat_analyzer/processing/thickness_estimator.py`
8. **Uniformity Analysis** → `esnf_mat_analyzer/analysis/uniformity_metrics.py`
9. **Gel Boundary Analysis** → `esnf_mat_analyzer/analysis/gel_boundary_analysis.py`
10. **Results Export** → `esnf_mat_analyzer/utils/data_exporter.py`

### Background Correction Architecture (Active Development Focus)
```
esnf_mat_analyzer/processing/background_correction/
├── __init__.py           # Unified interface for all approaches
├── advanced.py           # Primary 4-step scientific workflow (used by GUI)
├── mat_level.py          # Continuous mat regions (thick samples)
├── enhanced_scipy.py     # Research-grade with scipy optimization
└── enhanced_opencv.py    # Minimal dependencies (OpenCV only)
```

**Critical Pattern**: Each background correction class provides method selection via enum:
- `BackgroundCorrectionMethod.NONE` → No correction (preserves raw data)
- `BackgroundCorrectionMethod.POLYNOMIAL_SURFACE` → 2D polynomial fitting (recommended default)
- `BackgroundCorrectionMethod.LARGE_KERNEL_BLUR` → Morphological background estimation
- `BackgroundCorrectionMethod.COMPLETE_WORKFLOW` → Full 4-step workflow with optional homomorphic FFT

### New Modules (2026-01-29)

#### Synthetic Data Generation (`esnf_mat_analyzer/data/synthetic_mat_generator.py`)
- Generates ground-truth thickness maps for pipeline validation
- Patterns: uniform, radial_gradient, gaussian_spots, multi_frequency, realistic_mat, calibration_target
- Use for testing FFT filtering behavior and calibration accuracy

#### Frequency Diagnostics (`esnf_mat_analyzer/analysis/frequency_diagnostics.py`)
- Analyzes spatial frequency content before/after processing
- Identifies which real features are being removed by filtering
- Use when heatmaps appear "smoothed" or missing fine details

#### Three-Region Detection (`esnf_mat_analyzer/processing/region_detector.py`)
- Segments images into: background (collection surface), gel (dark perimeter), mat (bright deposition)
- Gel is darker than background (wet polymer absorbs light)
- Use for proper intensity calibration and gel influence studies

#### Intensity Calibration (`esnf_mat_analyzer/processing/calibration.py`)
- Converts intensity to normalized 0-100 thickness scale
- Black reference: background outside mat (0 = no thickness)
- White reference: saturation level OR max intensity (100 = max thickness)
- Optional physical calibration in micrometers with marked reference point

#### Gel Boundary Analysis (`esnf_mat_analyzer/analysis/gel_boundary_analysis.py`)
- Quantifies gel shape: circularity, irregularity, aspect ratio, solidity
- Correlates gel shape with mat uniformity (edge effects)
- Supports investigation of gel influence on deposition uniformity

## Entry Points & Interface Patterns

### Primary Entry Points
- **GUI**: `python run_esnf_analyzer.py` → `esnf_mat_analyzer/gui/main_window.py`
- **CLI**: `esnf-analyzer` command → `esnf_mat_analyzer/cli/main.py`
- **API**: `NanoFiberAnalyzer.from_config(config)` → `esnf_mat_analyzer/core/analyzer.py`

### Dependency Injection Pattern
```python
# Core creation pattern - used throughout
from esnf_mat_analyzer.main import setup_dependencies
from esnf_mat_analyzer.config.config_manager import get_default_config

config = get_default_config()
analyzer = setup_dependencies(config)  # Returns fully configured NanoFiberAnalyzer
```

### Configuration System
- **Type-safe dataclasses**: `esnf_mat_analyzer/core/data_types.py`
- **YAML/JSON loading**: `esnf_mat_analyzer/config/config_manager.py`
- **Enum-based method selection**: Background correction, thickness models, etc.

## FFT Enhancement (Critical - May Remove Real Features)

**Default: DISABLED** — FFT enhancement in visualization can remove real thickness variations.

Configuration in `VisualizationConfig`:
- `fft_enhancement_enabled: bool = False` — Master switch (keep False unless needed)
- `fft_blend_ratio: float = 0.0` — 0=no FFT, 1=full FFT (recommend ≤0.3 if enabled)
- `fft_sigma_divisor: float = 12.0` — Higher=gentler filtering, preserves more features

**When to enable FFT**: Only when strong illumination gradients obscure thickness variations
AND you've verified with synthetic data that real features aren't being removed.

**Validation workflow**:
```python
from esnf_mat_analyzer.data.synthetic_mat_generator import SyntheticMatGenerator
from esnf_mat_analyzer.analysis.frequency_diagnostics import FrequencyDiagnostics

# Generate known-frequency test pattern
gen = SyntheticMatGenerator()
result = gen.generate()  # Multi-frequency pattern

# Analyze what filtering removes
diag = FrequencyDiagnostics()
report = diag.generate_diagnostic_report(original, filtered, ground_truth=result.ground_truth_thickness)
```

## Development Workflows

### Testing Strategy
```bash
# Run organized test suite
pytest tests/unit/              # Fast unit tests
pytest tests/integration/       # Slower integration tests  
pytest tests/gui/              # GUI-specific tests
pytest -m "not slow"           # Skip computationally expensive tests
```

### Synthetic Data Validation Workflow
```python
# Generate validation suite
from esnf_mat_analyzer.data.synthetic_mat_generator import SyntheticMatGenerator, generate_fft_validation_set
from pathlib import Path

gen = SyntheticMatGenerator()
results = gen.generate_validation_suite(Path("test_data/synthetic"))
fft_results = generate_fft_validation_set(Path("test_data/synthetic"))
```

### Adding New Background Correction Methods
1. **Add enum value**: `BackgroundCorrectionMethod` in `data_types.py`
2. **Implement method**: In appropriate `background_correction/*.py` file
3. **Update GUI mapping**: Radio buttons in `gui/main_window.py` 
4. **Add configuration**: Update `config_manager.py` enum mappings
5. **Add tests**: Create test in `tests/integration/test_*_background_correction.py`

## Project-Specific Conventions

### File Organization Standards
- **Tests**: All in `tests/` with subdirectories (unit/, integration/, gui/, data/)
- **Scripts**: Debug/benchmark tools in `scripts/debug/`, `scripts/benchmarks/`
- **Examples**: Demo files in `examples/`
- **Documentation**: Guides in `docs/guides/`, images in `docs/images/`

### Import Path Patterns
```python
# Background correction imports
from esnf_mat_analyzer.processing.background_correction.advanced import AdvancedBackgroundProcessor

# New modules (2026-01-29)
from esnf_mat_analyzer.processing.region_detector import ThreeRegionDetector
from esnf_mat_analyzer.processing.calibration import IntensityCalibrator
from esnf_mat_analyzer.analysis.gel_boundary_analysis import GelBoundaryAnalyzer
from esnf_mat_analyzer.analysis.frequency_diagnostics import FrequencyDiagnostics
from esnf_mat_analyzer.data.synthetic_mat_generator import SyntheticMatGenerator

# Core components
from esnf_mat_analyzer.core.analyzer import NanoFiberAnalyzer
from esnf_mat_analyzer.core.data_types import Config, AnalysisResult, BackgroundCorrectionMethod

# CLI vs GUI patterns
from esnf_mat_analyzer.cli.main import main_cli        # Professional CLI with colors/progress
from esnf_mat_analyzer.gui.main_window import MainWindow  # Tkinter GUI
```

### Scientific Analysis Patterns
- **ROI Handling**: User ROI (rectangular) → Gel detection (optimal boundary) → Analysis mask
- **Scale Calibration**: Ruler detection provides `spatial_scale_pixels_per_mm`
- **Error Handling**: Graceful degradation with fallback methods for each step
- **Validation**: Quality metrics for each processing step (saturation rates, SNR, etc.)

## Testing & Validation Patterns

### Background Correction Testing Pattern
```python
# Standard validation metrics used throughout tests
def analyze_correction_quality(original, corrected, mask=None):
    metrics = {}
    metrics['saturation_pct'] = 100 * np.sum(corrected >= 254) / corrected.size
    metrics['dynamic_range'] = np.percentile(corrected, 98) - np.percentile(corrected, 2)
    metrics['signal_preservation'] = np.mean(corrected[mask]) if mask else 0
    return metrics
```

### GUI Testing Pattern
- **Integration tests**: Load actual images, test UI workflows
- **Method comparison**: Side-by-side visualization of all background methods
- **Configuration validation**: Ensure GUI selections update analysis config properly

## Key External Dependencies
- **OpenCV**: Image processing, background correction algorithms
- **scikit-image**: Advanced image analysis, morphological operations  
- **NumPy/SciPy**: Scientific computing, signal processing
- **Matplotlib**: Visualization, heatmap generation
- **tkinter**: GUI framework (included with Python)
- **Optional**: `colorama` (CLI colors), `tqdm` (progress bars)

## Common Debugging Patterns
- **Image processing issues**: Check `test_all_background_methods.py` for method comparison
- **Configuration problems**: Use `validate_config_file()` from config_manager
- **GUI integration**: Run `test_gui_background_integration.py` to verify UI workflow
- **Import errors**: Check recent reorganization - background correction moved to submodule

When working with this codebase, prioritize understanding the background correction pipeline as it's the primary development focus and affects all downstream analysis quality.

## Task Tracking (Hybrid Approach)

This project uses a **hybrid task tracking system** designed for AI assistant limitations (no persistent memory between sessions):

### Two Files, Two Purposes

1. **`CURRENT.md`** (Session Focus)
   - Active tasks for current development focus
   - Updated at **start** of each session (read it first!)
   - Updated at **end** of session with progress
   - Lightweight, 3-5 active tasks max
   - Contains: priorities, recently completed, blockers, session notes

2. **`task-ledger.md`** (Audit Trail)
   - Complete historical record of all tasks (T001, T002, etc.)
   - Formal tracking with IDs, statuses, prerequisites, completion dates
   - Updated when tasks are **completed** (move from CURRENT.md to ledger)
   - Reference for what was done and when

### Workflow for AI Assistant

**At Session Start:**
1. Read `CURRENT.md` to understand active priorities
2. Reference task IDs (T050, etc.) when they exist

**During Session:**
- Use informal numbered steps for planning (this is natural and fine)
- Work through CURRENT.md priorities in order
- Note any blockers or new tasks discovered

**At Session End (or when completing significant work):**
1. Update `CURRENT.md` - mark completed items, add new active tasks
2. Update `task-ledger.md` - move completed tasks to DONE with dates
3. Add any new tasks to ledger with next available ID

### Task Statuses (for ledger)
`TODO` | `IN-PROGRESS` | `BLOCKED` | `DONE` | `DEFERRED`

### Why This Works
- **CURRENT.md** is small enough to always fit in context
- In-conversation TODOs are acceptable for planning steps
- **task-ledger.md** provides the audit trail without being the active workflow
- No conflict between conversational planning and formal tracking

## Branch Naming Convention
When creating feature or fix branches, follow a numeric minor-increment policy derived from the current branch name when that branch uses a numeric suffix. Rules:

1. If the current branch matches the pattern `<name>_<major>` or `dev_<major>` (for example `dev_1`), create the new branch by incrementing the minor version: `dev_<major>.<minor>` where `<minor>` starts at `1` for the first child branch. Example: from `dev_1` → `dev_1.1`.
2. If multiple child branches already exist, pick the next unused minor (e.g., `dev_1.1`, `dev_1.2`, ...).
3. Append a short, descriptive suffix after the version separated by a dash, e.g. `dev_1.1-add-ledger`, `feature-2.3-fix-anisotropy`.
4. If the current branch does not use a numeric suffix, fall back to `feature/<short-desc>` or `fix/<short-desc>`.
5. Update `.github/copilot-instructions.md` when modifying this policy.
