# Background Correction Methods in ESNF Mat Analyzer GUI

## Overview

The ESNF Mat Analyzer GUI now includes advanced background correction capabilities that help you get better results when analyzing electrospun nanofiber mats. This guide explains how to use the new background correction features.

## Features Added

### 1. Background Correction Method Selection
- **Radio Buttons**: Choose from 5 different background correction methods
- **Preview Button**: Compare all methods side-by-side before choosing
- **Real-time Updates**: Your selection applies immediately to analysis

### 2. Available Background Correction Methods

| Method | Description | Best For |
|--------|-------------|----------|
| **None** | No background correction (original behavior) | Clean images with uniform background |
| **Basic** | Robust statistical correction using percentile-based normalization | Most general cases, good default |
| **Rolling Ball** | 3D rolling ball algorithm for uneven illumination | Gradient backgrounds, lighting variations |
| **RESTORE** | Iterative deconvolution method | High-frequency noise, complex patterns |
| **Homomorphic** | Frequency domain filtering | Multiplicative noise, illumination issues |

## How to Use

### Step 1: Load Your Image
1. Click "Browse" to select an ESNF mat image
2. The image will appear in the main canvas
3. Use the zoom and pan controls to examine the image

### Step 2: Select ROI (Region of Interest)
1. Click and drag to select the area you want to analyze
2. The ROI will be highlighted with a red rectangle
3. You can adjust the ROI by dragging the corners

### Step 3: Choose Background Correction Method

#### Option A: Quick Selection
1. Use the radio buttons to select a method:
   - Start with "Basic" for most images
   - Try "Rolling Ball" if you have uneven lighting
   - Use "None" only if your background is already uniform

#### Option B: Preview and Compare
1. Click the "Preview Methods" button
2. A new window will open showing all 5 methods applied to your image
3. Each preview shows:
   - The corrected image
   - Statistics (min, max, mean values)
   - Saturation percentage
4. Click "Select" under the method that looks best
5. The preview window will close and your selection will be applied

### Step 4: Run Analysis
1. Ensure "Background Leveling" is checked (recommended)
2. Click "Analyze"
3. The analysis will use your selected background correction method
4. Results will appear in a new window

## Tips for Best Results

### For Images with White ESNF Accumulation
- **Problem**: White fiber areas not showing up in heatmap
- **Solution**: Use "Basic" or "Rolling Ball" methods
- **Why**: These methods properly normalize the image so white = high thickness

### For Uneven Lighting
- **Problem**: One side of image brighter than the other
- **Solution**: Use "Rolling Ball" method
- **Why**: Rolling ball correction handles gradual illumination changes

### For Noisy Images
- **Problem**: Lots of small-scale noise or artifacts
- **Solution**: Try "RESTORE" or "Homomorphic" methods
- **Why**: These methods can reduce different types of noise

### For Clean, Well-Lit Images
- **Problem**: Image already looks good
- **Solution**: Use "Basic" method or "None"
- **Why**: Minimal processing preserves image quality

## Understanding the Preview Window

When you click "Preview Methods", you'll see 5 panels:

### Image Statistics Explained
- **Original**: Shows min-max range and mean of original image
- **Corrected**: Shows min-max range and mean after correction
- **Saturation**: Percentage of pixels that are very bright (≥240)

### What to Look For
1. **Good contrast**: Wide range between min and max values
2. **Balanced histogram**: Mean value around 100-150
3. **Low saturation**: Less than 5% saturation is ideal
4. **Visual quality**: Fibers should be clearly visible

### Selecting the Best Method
- Choose the method where:
  - White fiber areas are clearly visible
  - Background is uniform
  - Statistics show good contrast
  - Saturation is reasonable (<10%)

## Configuration

### Settings that Affect Background Correction
- **Background Leveling**: Should be enabled for best results
- **Auto Range Heatmap**: Adjusts color mapping automatically

### Advanced Configuration
For advanced users, you can modify the background correction parameters by editing the configuration YAML file:

```yaml
processing:
  background_correction_method: BASIC  # Options: NONE, BASIC, ROLLING_BALL, RESTORE, HOMOMORPHIC
  leveling:
    enabled: true
    saturation_threshold_percentile: 99.5
```

## Troubleshooting

### Problem: Preview window shows errors
- **Solution**: Make sure an image is loaded and ROI is selected
- **Check**: Verify the image file isn't corrupted

### Problem: All methods look the same
- **Possible cause**: Image already has good background
- **Solution**: Use "None" or "Basic" method

### Problem: Analysis fails after selecting method
- **Check**: Ensure ROI is properly selected
- **Try**: Restart the application and try again

### Problem: Results don't match expectations
- **Try**: Different background correction method
- **Check**: ROI selection covers the right area
- **Verify**: Background leveling is enabled

## Technical Notes

### Performance
- Preview generation may take a few seconds for large images
- Rolling Ball method is the slowest but most thorough
- Basic method offers the best speed/quality balance

### Memory Usage
- Large images may require significant memory for preview
- Close preview windows when done to free memory

### Supported Image Formats
- PNG, JPEG, TIFF (8-bit and 16-bit)
- RGB and grayscale images
- Various file extensions: .jpg, .jpeg, .png, .tif, .tiff

## Example Workflow

1. **Load Image**: Select your ESNF mat image
2. **Set ROI**: Draw rectangle around analysis area
3. **Preview Methods**: Click "Preview Methods" button
4. **Compare Results**: Look at all 5 correction methods
5. **Select Best**: Click "Select" under the best-looking result
6. **Run Analysis**: Click "Analyze" with your chosen method
7. **Review Results**: Examine the thickness heatmap and metrics

## Need Help?

If you encounter issues or need assistance:
1. Check that all requirements are installed
2. Verify your image is supported
3. Try different background correction methods
4. Consult the main README.md for general troubleshooting

The background correction feature significantly improves the accuracy of ESNF mat analysis, especially for images with challenging lighting or background conditions.
