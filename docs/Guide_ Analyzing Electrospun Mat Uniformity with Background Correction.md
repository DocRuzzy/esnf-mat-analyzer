# **Guide: Analyzing Electrospun Mat Uniformity with Background Correction**

This document outlines a scientifically robust procedure to analyze the uniformity of electrospun mats from images with non-standardized, uneven illumination. The core principle is to computationally model the background lighting from the area surrounding the mat and use this model to correct the image before analysis.

This method allows for the objective, quantitative comparison of mat uniformity, even when a proper flat-field correction was not performed during image acquisition.

### **Step 1: Isolate the Background Pixels**

The first objective is to create a digital mask that isolates the background pixels. This allows the software to learn the illumination pattern without being confused by the mat or the hydrogel.

1. **Convert to Grayscale:** All analysis should be performed on a single-channel grayscale image (e.g., 8-bit, with pixel values from 0 to 255). This removes color, which is not relevant to thickness, and simplifies calculations.  
2. **Create a "Foreground Mask":**  
   * **Mask the Mat:** The mat is the brightest object in the image. Apply a high-pass intensity threshold (e.g., select all pixels with a value \> 180\) to create a binary mask that cleanly selects the mat.  
   * **Mask the Hydrogel:** The hydrogel is the darkest object. Apply a low-pass intensity threshold (e.g., select all pixels with a value \< 50\) to create a mask for the hydrogel.  
   * **Combine:** Merge the mat mask and the hydrogel mask. This combined mask represents the entire "foreground" of your image—the areas that are not part of the background.  
3. **Create the Final Background Mask:**  
   * Invert the "Foreground Mask". The resulting image is your final **Background Mask**. In this mask, only the pixels corresponding to the black polymer sheet are white (active), while the areas containing the mat and hydrogel are blacked out (inactive).

### **Step 2: Model the Uneven Illumination**

Using only the background pixels identified in Step 1, you will now generate a smooth surface that estimates the lighting variation across the entire image frame.

#### **Method A: Polynomial Surface Fitting (Most Robust)**

This method fits a mathematical surface to the background pixel intensities. It is highly accurate for modeling smooth lighting gradients.

* **Concept:** A 2D polynomial function is fitted to the intensity values of the background pixels. The function takes pixel coordinates (x, y) as input and outputs the estimated background intensity at that location.  
* **Function:** The general form is Intensity(x,y)=p00​+p10​x+p01​y+p20​x2+p11​xy+p02​y2+...  
* **Implementation:** Use a function (available in MATLAB, Python with Scikit-learn, or ImageJ plugins) to fit a 2nd or 3rd-order polynomial to the background pixels. Then, use this function to generate a new image representing the smooth background across the entire frame.

#### **Method B: Large-Kernel Blurring (Simpler Alternative)**

This method uses a simpler, filter-based approach that is often sufficient and computationally faster.

* **Concept:** A large blurring filter is applied to the background-only image, effectively averaging the known background pixels to "fill in" the unknown area under the mat.  
* **Implementation:**  
  1. Create an image where the background pixels have their original intensity and the foreground (mat/hydrogel) is black (value 0).  
  2. Apply a Gaussian blur filter. The radius (kernel size) of the blur must be significantly larger than the diameter of the mat. This ensures a smooth interpolation across the masked-out region.

The output of either method is a new image: your **Estimated Background**. It will appear as a smooth, continuous gradient that models the uneven illumination.

### **Step 3: Correct the Image**

With the Estimated Background model, you can now level the illumination in the original image. For correcting multiplicative effects like lighting, division is the physically correct mathematical operation.

1. **Perform Division:** The correction is performed on a pixel-by-pixel basis.  
   * Corrected Value \= Original Value / Estimated Background Value  
2. **Normalize for Viewing:** The result of the division will be floating-point numbers, often centered around 1.0. To convert this back to a standard image format (like 8-bit grayscale), you should normalize it.  
   * Final Image Value \= (Corrected Value) \* (Mean Intensity of Estimated Background)

The Final Image will show your mat on a flat, uniform gray background. The lighting artifacts should be almost entirely removed.

### **Step 4: Analyze Mat Uniformity**

You can now confidently analyze the mat from the corrected image.

1. **Isolate the Mat:** Use the original mat mask you created in Step 1 to select only the mat pixels from your Final Image.  
2. **Quantify Uniformity:** For the population of pixels within the mat, calculate the following statistics. These intensity values now serve as a reliable proxy for thickness.  
   * **Mean Intensity:** The average "thickness" of the mat.  
   * **Standard Deviation (SD):** A measure of the absolute variation in thickness.  
   * Coefficient of Variation (CV): This is your primary metric for uniformity. It normalizes the standard deviation by the mean, allowing for fair comparison between different samples.  
     CV=MeanStandard Deviation​

     A lower CV indicates a more uniform mat.  
3. **Visualize the Results:**  
   * **Heatmap:** Display the isolated mat pixels using a color map (e.g., "jet" or "viridis"). This provides an immediate, intuitive visualization of thicker (hotter colors) and thinner (cooler colors) regions.  
   * **3D Surface Plot:** Create a 3D plot where the X and Y axes are the pixel coordinates and the Z axis is the corrected intensity value. This renders the mat's "topography," making variations easy to see.