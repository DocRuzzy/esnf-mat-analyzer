# GUI Improvements: Background Leveling and Heatmap Range Control

## New Features Added

### 1. Background Leveling Control
- **Location**: Analysis panel in the GUI
- **Feature**: Checkbox labeled "Enable Background Leveling"
- **Default**: Enabled (checked)
- **Purpose**: Allows users to enable or disable background leveling during image preprocessing
- **Impact**: When disabled, the raw image intensities are used without background correction

### 2. Heatmap Range Control
- **Location**: Analysis panel in the GUI
- **Feature**: Checkbox labeled "Auto-adjust Heatmap Range"
- **Default**: Enabled (checked)
- **Purpose**: Controls how the color range is set for thickness heatmaps
- **Impact**: 
  - **Enabled**: Uses 2nd-98th percentile range for better contrast of the main data
  - **Disabled**: Uses full data range including outliers

## Technical Details

### Background Leveling Configuration
```python
# In ProcessingConfig
leveling: LevelingConfig = field(default_factory=LevelingConfig)

# LevelingConfig structure
@dataclass
class LevelingConfig:
    enabled: bool = True        # Enable/disable background leveling
    kernel_size: int = 25       # Morphological operation kernel size
```

### Heatmap Range Configuration
```python
# New fields in VisualizationConfig
heatmap_percentile_range: Tuple[float, float] = (2.0, 98.0)  # Percentile range
auto_range_heatmap: bool = True                               # Auto-range enable/disable
```

## Configuration File Updates

### YAML Configuration
```yaml
processing:
  leveling:
    enabled: true
    kernel_size: 25

visualization:
  auto_range_heatmap: true
  heatmap_percentile_range: [2.0, 98.0]
```

## Benefits

### Background Leveling Control
1. **Flexibility**: Users can compare results with and without background correction
2. **Debugging**: Helps identify if background leveling is causing issues
3. **Performance**: Can be disabled for faster processing if not needed

### Improved Heatmap Range
1. **Better Visualization**: Main fiber mat data is displayed with optimal contrast
2. **Outlier Handling**: Extreme values don't dominate the color scale
3. **Customizable**: Percentile range can be adjusted in configuration
4. **User Control**: Can be toggled on/off via GUI checkbox

## Before vs After

### Before
- Background leveling was always enabled with no user control
- Heatmap color range included all values, often leading to poor contrast
- Outliers could make the main data appear washed out

### After
- ✅ User can enable/disable background leveling via checkbox
- ✅ Heatmap range automatically excludes outliers for better contrast
- ✅ User can toggle auto-range on/off for comparison
- ✅ Configuration is preserved and customizable

## Usage Instructions

1. **Launch the GUI**: Run `python run_gui_app.py`
2. **Load an image**: Use "Select Files" to choose your image
3. **Select ROI**: Draw a rectangle around the region to analyze
4. **Configure processing**:
   - Check/uncheck "Enable Background Leveling" as needed
   - Check/uncheck "Auto-adjust Heatmap Range" as needed
5. **Run analysis**: Click "Analyze"
6. **View results**: Click "Show Heatmap" to see the thickness map

## Testing
Run the test script to verify all improvements:
```bash
python test_improvements.py
```
