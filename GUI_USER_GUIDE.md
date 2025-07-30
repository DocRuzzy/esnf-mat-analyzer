# GUI User Guide

## Nanofiber Analyzer GUI

The GUI application provides an intuitive interface for analyzing nanofiber thickness uniformity from images.

### Features

#### Image Display
- **Automatic Scaling**: Large images are automatically scaled to fit the display window
- **Zoom Controls**: 
  - Zoom In (+) and Zoom Out (-) buttons
  - Mouse wheel zoom
  - Fit to Window button (fits entire image in view)
  - 100% button (shows image at actual size)
- **Pan Navigation**: Right-click and drag to pan around large images
- **Scrollbars**: Horizontal and vertical scrollbars for navigation

#### File Management
- Select multiple image files at once
- Preview images by clicking on them in the file list
- Supports common image formats: PNG, JPG, JPEG, BMP, GIF, TIFF

#### ROI Selection
- Left-click and drag to select Region of Interest (ROI)
- ROI coordinates are automatically converted from display coordinates to original image coordinates
- Red rectangle shows the selected area

#### Analysis
- Click "Analyze" to process the selected image with the current ROI
- Results are displayed in a separate window showing metrics and processing information

### Usage Instructions

1. **Start the Application**:
   ```bash
   python test_gui.py
   ```
   or
   ```bash
   python -m esnf_mat_analyzer.gui.main_window
   ```

2. **Load Images**:
   - Click "Select Files" to choose image files
   - Click on any file in the list to preview it

3. **Navigate the Image**:
   - Use zoom controls to adjust the view
   - Right-click and drag to pan around the image
   - Use mouse wheel for quick zooming

4. **Select ROI**:
   - Left-click and drag on the image to select the analysis region
   - The red rectangle shows your selection

5. **Analyze**:
   - Click "Analyze" to process the image
   - Results will appear in a new window

### Controls Reference

- **Left Mouse Button**: Select ROI (drag to create rectangle)
- **Right Mouse Button**: Pan image (drag to move view)
- **Mouse Wheel**: Zoom in/out
- **+ Button**: Zoom in
- **- Button**: Zoom out  
- **Fit Button**: Scale image to fit window
- **100% Button**: Show image at actual size

### Technical Details

- The GUI automatically handles coordinate conversion between display and original image coordinates
- Images are scaled using high-quality Lanczos resampling
- Zoom range: 10% to 500%
- ROI coordinates are validated to ensure they stay within image bounds
