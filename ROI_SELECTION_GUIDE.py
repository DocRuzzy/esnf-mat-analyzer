"""
ROI Selection Guidelines for Optimal Multiscale Analysis

Use these guidelines when selecting ROI in the GUI to avoid perfect uniformity scores:

1. AVOID EDGES: Don't include image edges which often contain uniform backgrounds
2. AVOID RULERS: Exclude ruler areas which are very uniform 
3. CENTER FOCUS: Select the central 50-70% of the image
4. MAT REGIONS: Ensure ROI covers actual fiber mat, not background
5. SIZE MATTERS: Use ROI of at least 300x300 pixels for meaningful wavelet analysis
"""

# Recommended ROI calculation for different image sizes:

def get_recommended_roi(image_width, image_height):
    """Calculate a recommended ROI that avoids common uniformity issues."""
    
    # Use 30% margins from edges (focus on central 70%)
    margin_x = int(image_width * 0.15)   # 15% margin on each side
    margin_y = int(image_height * 0.15)  # 15% margin top/bottom
    
    roi_x = margin_x
    roi_y = margin_y
    roi_width = image_width - 2 * margin_x
    roi_height = image_height - 2 * margin_y
    
    # Ensure minimum size for wavelet analysis
    min_size = 300
    if roi_width < min_size or roi_height < min_size:
        # Fall back to smaller margins if image is small
        margin_x = max(50, int(image_width * 0.1))
        margin_y = max(50, int(image_height * 0.1))
        
        roi_x = margin_x
        roi_y = margin_y
        roi_width = image_width - 2 * margin_x
        roi_height = image_height - 2 * margin_y
    
    return (roi_x, roi_y, roi_width, roi_height)

# Example for your image (1816x2420):
# Recommended ROI: (272, 363, 1272, 1694)
# This covers the central 70% and avoids edge artifacts
