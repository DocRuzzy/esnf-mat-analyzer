

# **A First-Principles Approach to the Non-Destructive Analysis of Electrospun Mat Thickness and Uniformity via Digital Image Processing**

## **Section 1: Foundational Optical Principles for Characterizing Scattering Nanofiber Mats**

To develop a robust, science-based methodology for analyzing the thickness and uniformity of electrospun nanofiber mats, it is imperative to first establish the fundamental optical principles that govern the interaction of light with these complex materials. The challenges presented by a reflective substrate and non-uniform illumination necessitate a first-principles approach, beginning with an understanding of how the captured image relates to the physical properties of the mat. This section deconstructs the optical phenomena at play, justifies the selection of an appropriate theoretical framework, and lays the groundwork for the empirical calibration strategy that forms the core of this analysis protocol.

### **1.1 The Illumination-Reflectance Model: Deconstructing the Image**

At its most fundamental level, a digital image is a two-dimensional representation of light intensity captured by a sensor. The intensity value, I, at any given pixel coordinate (x,y) is not solely a function of the object being imaged but is rather a product of two distinct components: the illumination incident on the scene, L(x,y), and the reflectance properties of the object itself, R(x,y). This relationship is described by the illumination-reflectance model 1:

I(x,y)=L(x,y)×R(x,y)  
Here, L(x,y) represents the intensity and spatial distribution of the light source. In an ideal scenario, illumination is perfectly uniform across the entire scene, making L(x,y) a constant. However, in nearly all practical laboratory settings, L(x,y) is a spatially varying function due to the directionality of light sources, distance fall-off, and ambient reflections. This non-uniformity manifests as the smooth brightness gradients observed in the provided sample images.

The second component, R(x,y), represents the fraction of incident light that is reflected by the object at each point. This reflectance is an intrinsic property of the material and is directly related to its composition, morphology, and, critically for this analysis, its thickness. The primary objective of this entire protocol is to computationally isolate and analyze R(x,y) to deduce the thickness of the nanofiber mat. The core challenge articulated in the user query—the need to correct for a "non-flat" background—is precisely the problem of separating the influence of the non-uniform illumination component, L(x,y), from the desired reflectance component, R(x,y). Any quantitative analysis that fails to account for the multiplicative nature of L(x,y) will incorrectly attribute changes in brightness caused by lighting gradients to variations in mat thickness, leading to significant and systematic errors.

### **1.2 The Unique Optical Nature of Electrospun Mats: A Scattering-Dominated System**

Electrospun nanofiber mats possess a unique microstructure that dictates their optical behavior. These materials are composed of polymer fibers with diameters typically in the range of tens to hundreds of nanometers, resulting in an extremely high surface-area-to-volume ratio and a highly porous, interconnected structure.2 Optically, these mats appear white and opaque not because of high specular (mirror-like) reflection, but due to intense, multiple light scattering events occurring at the vast number of fiber-air interfaces.4

When light enters the mat, it is deflected in myriad directions by the dense, irregularly distributed nanofibers. This process, known as diffuse scattering, randomizes the direction of the incident photons. The "whiteness" observed is the result of this diffusely scattered light being remitted from the mat's surface across a broad range of angles. The efficiency of this scattering process is influenced by several structural parameters, including:

* **Fiber Diameter and Distribution:** The size of the fibers relative to the wavelength of light is a key factor in scattering efficiency.4  
* **Porosity and Pore Size:** The density of fiber packing and the size of the voids between fibers determine the number of scattering interfaces a photon will encounter as it travels through the mat.2  
* **Mat Thickness:** As the thickness of the mat increases, a photon must travel a longer path and undergo more scattering events, which increases the probability that it will be scattered back out of the surface rather than being transmitted through to the black substrate.  
* **Defects and Fiber Orientation:** Structural defects, fiber alignment, and internal strain can also alter the local optical properties of the mat.4

Because the appearance of the mat is governed by this complex, scattering-dominated process, simple optical models that are based primarily on absorption or specular reflection are insufficient. The relationship between the mat's apparent brightness (its diffuse reflectance) and its physical thickness is non-trivial and requires a theoretical framework capable of describing light transport in such a turbid, or highly scattering, medium.

### **1.3 Theoretical Frameworks for Light Interaction in Turbid Media**

To model the relationship between mat thickness and observed intensity, one must turn to theories developed for turbid media. Two such frameworks are particularly relevant: the Modified Beer-Lambert Law and Kubelka-Munk Theory.

#### **1.3.1 The Modified Beer-Lambert Law (MBLL)**

The classical Beer-Lambert law describes the attenuation of light as it passes through a non-scattering, absorbing medium. It is not applicable to turbid systems like nanofiber mats where scattering is dominant. The **Modified Beer-Lambert Law (MBLL)** extends this principle to scattering media by introducing a term to account for the increased optical path length that photons travel due to multiple scattering events.8 The differential form of the MBLL relates the change in measured light attenuation,

ΔA, to the change in the medium's absorption coefficient, Δμa​, and the total mean path length of detected photons, L 8:

ΔA=L⋅Δμa​  
This relationship can be further expressed in terms of the Differential Pathlength Factor (DPF), which is the ratio of the mean pathlength to the physical distance between the light source and detector.10 The MBLL has been successfully used to develop real-time thickness measurement systems for electrospun mats based on measuring light

*transmittance* during the spinning process.11

However, the MBLL is most directly and easily applied in a transmission geometry, where the light source and detector are on opposite sides of the sample. The present application involves a reflection geometry, where the light source and camera are on the same side. While the underlying physics of scattering-induced pathlength increase is still valid, applying the MBLL directly would be complex. A more suitable framework exists for describing diffuse reflectance.

#### **1.3.2 Kubelka-Munk (K-M) Theory: The Cornerstone for Diffuse Reflectance**

The most appropriate theoretical framework for this application is the **Kubelka-Munk (K-M) theory**. Originally developed to describe the appearance of paint films, K-M theory is a two-flux radiative transfer model specifically designed for light-scattering layers such as paints, textiles, and paper—all materials that are optically analogous to a nanofiber mat.12

The theory models the light within the scattering layer as two diffuse fluxes: one traveling downward (into the material) and one traveling upward (out of the material). It describes the change in these fluxes as a function of two fundamental material properties:

* **The K-M absorption coefficient, K:** Represents the light absorption per unit path length.  
* **The K-M scattering coefficient, S:** Represents the light scattering (specifically, back-scattering) per unit path length.

Using these coefficients, K-M theory provides an equation for the diffuse reflectance, R, of a material of thickness X placed over a substrate with reflectance Rg​.13 A key prediction of the theory is the concept of the reflectance of an infinitely thick layer, denoted as

R∞​. This is the maximum possible reflectance a material can achieve, at which point adding more thickness does not increase its brightness.12 The relationship is given by the Kubelka-Munk function:

F(R∞​)=2R∞​(1−R∞​)2​=SK​  
This equation highlights that the maximum brightness of the mat is determined by the ratio of its intrinsic absorption to its scattering properties. As the mat thickness X increases from zero, its reflectance R will non-linearly approach this saturation value R∞​. This non-linear, saturating behavior is the critical physical phenomenon that must be captured in any model relating image intensity to mat thickness.

### **1.4 Strategic Pivot: From Theoretical Modeling to Empirical Calibration**

The physics of light interaction with the nanofiber mat is demonstrably complex. While Kubelka-Munk theory provides a powerful and appropriate theoretical foundation, a purely first-principles application of the model would present significant practical challenges. It would require the independent measurement of the intrinsic K-M scattering (S) and absorption (K) coefficients for the specific polymer and nanofiber morphology being produced.15 These parameters are not readily available, can be difficult to measure directly, and may even vary slightly with changes in electrospinning process parameters. Furthermore, the K-M theory itself has known limitations and assumes ideal diffuse illumination and isotropic scattering, conditions which may not be perfectly met in a real-world setup.12

Attempting to solve the K-M equations from first principles would be an arduous and potentially inaccurate endeavor. A more robust, practical, and scientifically sound strategy is to leverage the fundamental conclusion of the theory without being constrained by its practical application challenges. The K-M model proves that a definitive, albeit complex and non-linear, relationship exists between the physical thickness of the mat and its diffuse reflectance (the "whiteness" captured by the camera).

Therefore, the most effective path forward is to determine this relationship **empirically**. By creating a set of calibration standards—nanofiber mats of known, independently verified thicknesses—and measuring their corresponding reflectance (i.e., image intensity) under controlled conditions, one can construct a calibration curve. This curve becomes a direct, empirical model that maps observed intensity to physical thickness, effectively bypassing the need to explicitly determine the material-dependent S and K coefficients. This empirical calibration approach is scientifically justified by the complex, scattering-dominated nature of the material and forms the central pillar of the measurement protocol detailed in the subsequent sections.

## **Section 2: A Rigorous Protocol for Image Acquisition and Illumination Correction**

The foundation of any accurate image-based analysis is the acquisition of high-quality, reproducible data. For the task of correlating image intensity with mat thickness, it is paramount to control the imaging environment and computationally correct for inherent optical artifacts. The non-uniform illumination and reflective substrate are the primary challenges to overcome. This section outlines a systematic protocol for optimizing the image acquisition setup and details the gold-standard method for illumination correction, flat-fielding, as well as a powerful computational alternative. Adherence to this protocol is essential for the validity and accuracy of all subsequent analysis steps.

### **2.1 Optimizing the Imaging Environment for Consistency and Accuracy**

Before any images are captured, the physical setup must be configured to maximize signal quality and minimize artifacts. The goal is to create a stable and controlled environment where the only significant variable changing between images is the sample itself.

* **Lighting:** The ideal illumination for this application is diffuse, non-directional, and stable. Direct, harsh lighting will create strong specular reflections and deep shadows, which can obscure the subtle intensity variations related to mat thickness. An excellent solution is to use a photographic light tent or a dome illuminator with an integrating sphere. These devices scatter the light from the source, providing even, multi-directional illumination that minimizes specular glare. The light source itself (e.g., LED or halogen lamp) should be powered by a stabilized supply and allowed to warm up to a steady-state output before any images are acquired.  
* **Polarization:** A critical technique for this specific application is **cross-polarization**. Specular (mirror-like) reflections occur when light bounces off a smooth surface without being scattered. Both the surface of the black polymer substrate and the surfaces of the individual nanofibers can produce such reflections. This specularly reflected light does not carry information about the bulk scattering properties of the mat and thus acts as noise. To eliminate it, a polarizing filter is placed over the light source, and a second polarizing filter is placed over the camera lens. The two filters are oriented perpendicular (at 90 degrees) to each other. Light from the source becomes polarized. When this polarized light reflects specularly from a surface, its polarization state is largely preserved. The second, crossed polarizer on the camera lens will therefore block it. In contrast, light that enters the nanofiber mat and is multiply scattered becomes depolarized. A significant portion of this diffusely scattered light will be able to pass through the second polarizer and reach the camera sensor. This technique effectively isolates the desired diffuse reflectance signal, which contains the thickness information, from the unwanted specular glare.16  
* **Camera Settings:** To ensure comparability between images, the camera must be operated in full manual mode. Automatic settings will adjust exposure based on the scene's brightness, invalidating any attempt to correlate intensity with thickness. The following parameters must be fixed for all images, including calibration standards 17:  
  * **ISO:** Set to the lowest possible value to minimize sensor noise.  
  * **Aperture (f-stop):** Set to a value that provides sufficient depth of field to keep the entire mat surface in focus (e.g., f/8 or f/11).  
  * **Shutter Speed:** Adjusted to achieve a proper exposure where the brightest parts of the white mat are well-lit but not saturated (i.e., no pixels have the maximum possible intensity value).  
  * **White Balance:** Set to a fixed value (e.g., "Daylight" or a custom kelvin temperature) appropriate for the light source. Do not use "Auto White Balance."  
  * **Focus:** Use manual focus to ensure the plane of focus is consistently on the surface of the mat.  
* **Geometry:** The geometric relationship between the camera, light source, and sample must remain constant. The camera should be mounted on a copy stand or tripod to ensure it is perfectly perpendicular (normal) to the sample substrate. The distance from the camera to the sample must be identical for all acquired images. This prevents changes in magnification or perspective from affecting the measurements.

### **2.2 Primary Correction via Flat-Fielding: The Experimental Gold Standard**

Even with optimized lighting, no imaging system is perfect. Minor variations in the illumination field and imperfections in the optical path (e.g., dust on the sensor or lens) will persist. **Flat-field correction** is the standard scientific method to computationally remove these fixed-pattern artifacts.18 It is an experimental technique that directly measures the system's response to a uniform field.

#### **2.2.1 Procedure**

The procedure requires capturing three types of images under the exact same, fixed camera settings established in the previous section:

1. **Capture the Sample Image (Iraw​):** This is the standard image of the electrospun mat on the black polymer substrate.  
2. **Capture the Bright-Field Image (Ibright​):** This is an image of the background *without* the sample. For this application, this means an image of the clean, empty black polymer substrate. This image serves as a map of the non-uniform illumination pattern and captures any shadows or vignetting caused by the optical system.  
3. **Capture the Dark-Field Image (Idark​):** This is an image taken with the lens cap on, using the same ISO and shutter speed as the sample image. This image captures the camera's inherent electronic noise, including dark current (thermal noise) and bias signal (readout noise).

#### **2.2.2 Calculation**

The corrected image, Icorrected​, is then calculated on a pixel-by-pixel basis using the following formula 18:

Icorrected​=(Ibright​−Idark​)(Iraw​−Idark​)​×Mean(Ibright​−Idark​)  
The subtraction of Idark​ from both the raw and bright-field images removes the baseline electronic noise. The division of the noise-subtracted sample image by the noise-subtracted bright-field image corrects for the multiplicative illumination non-uniformity. A pixel in Iraw​ that is artificially bright due to a hotspot in the illumination will be divided by a correspondingly bright pixel in Ibright​, normalizing its value. The final multiplication by the mean of the corrected bright-field is a scaling step to restore the image's intensity to a familiar range, preventing it from being purely fractional values between 0 and 1\.

The result of this operation is an image where the background is rendered uniformly flat, and the intensity variations are, in principle, due only to the reflectance properties of the nanofiber mat itself. This is the most physically accurate method for background correction.

### **2.3 Computational Background Modeling: A Powerful Alternative**

In situations where acquiring a separate bright-field image for every imaging session is impractical, a computational approach can be used to estimate and remove the non-uniform background from the sample image itself. This method is predicated on the observation that the background illumination is a smooth, low-frequency variation across the image, while the features of interest (the mat texture) are higher-frequency.

#### **2.3.1 Method: 2D Polynomial Surface Fitting**

One of the most effective and widely used methods for computational background correction is fitting a mathematical surface to the background intensity.20

1. **Sample Background Points:** The first step is to identify points in the image that belong exclusively to the background. In the provided samples, these are the regions of the black polymer substrate outside the dark hydrogel frame. These points can be selected manually by the user or automatically using a preliminary segmentation step.  
2. **Fit a Surface:** A least-squares algorithm is used to fit a 2D polynomial function to the intensity values of the sampled background points. The function takes the form z=f(x,y), where (x,y) are the pixel coordinates and z is the predicted background intensity. A low-order polynomial is typically sufficient. For example:  
   * **1st Order (Plane):** z=ax+by+c  
   * 2nd Order (Quadratic): z=ax2+by2+cxy+dx+ey+f  
     The choice of order depends on the complexity of the illumination gradient; a simple gradient can be modeled with a plane, while more complex vignetting may require a quadratic or cubic surface.21  
3. **Generate Background Image:** A full-size image, Ibackground\_model​, is generated by evaluating the fitted polynomial function at every pixel coordinate (x,y) in the image. This creates a smooth surface that represents the estimated non-uniform illumination.  
4. **Correct the Image:** The modeled background is then removed from the raw image. While subtraction (Icorrected​=Iraw​−Ibackground\_model​) is common, division (Icorrected​=Iraw​/Ibackground\_model​) is often more physically correct, as it aligns with the multiplicative nature of the illumination-reflectance model.

This technique is well-established for correcting vignetting and other illumination artifacts in a variety of scientific imaging applications and provides a robust alternative when experimental flat-fielding is not performed.18

The choice between flat-fielding and computational modeling represents a trade-off. Flat-field correction is experimentally more rigorous and provides a direct physical measurement of the illumination artifacts, making it the preferred method for the highest quantitative accuracy. Computational modeling, particularly 2D polynomial fitting, is a powerful and often sufficient post-processing alternative that relies on the assumption that the background can be accurately represented by a smooth mathematical function. For the most rigorous scientific work, flat-field correction should be the primary strategy (Plan A). Polynomial fitting serves as an excellent and highly effective backup (Plan B), especially given the smooth, gradient-like illumination apparent in the sample images.

## **Section 3: Advanced and Alternative Background Correction Techniques**

While the methods of flat-fielding and polynomial surface fitting described in the previous section are the primary and most recommended approaches, a comprehensive protocol should include alternative techniques that may offer advantages in specific scenarios. These advanced computational methods can be particularly useful when the image background is difficult to sample (e.g., if the mat covers most of the frame) or when a more sophisticated separation of texture and illumination is required. This section explores two such methods: morphological background estimation and frequency-domain homomorphic filtering.

### **3.1 Morphological Background Estimation**

Morphological image processing is a collection of non-linear operations that modify image geometry by probing it with a small shape called a "structuring element".22 Morphological background estimation operates on the principle that the background illumination consists of large, smooth features, whereas the object of interest (the nanofiber mat) is composed of smaller, more textured features.

#### **3.1.1 Method: "Rolling Ball" or Morphological Opening**

The most common morphological technique for background subtraction is conceptually known as the "rolling ball" algorithm, which is implemented in popular image analysis software like ImageJ via its Subtract Background function.18 The algorithm can be understood as rolling a large 2D ball (or, more accurately, a paraboloid) underneath the 3D intensity landscape of the image. The surface traced out by the top of this ball as it rolls through the valleys of the intensity landscape forms the estimated background. Because the "ball" is chosen to be much larger than the bright features of the mat, it cannot fit into the "crevices" of the mat's texture, effectively smoothing over them and capturing only the underlying low-frequency background variation.

A closely related and more formally defined operation is **morphological opening**. An opening operation consists of an erosion followed by a dilation 22:

1. **Erosion:** This operation scans the image with a structuring element (e.g., a disk). At each position, the value of the central pixel in the output image is set to the *minimum* value of the pixels in the input image that are covered by the structuring element. When applied to a bright object on a dark background, erosion shrinks the bright object from its edges.  
2. **Dilation:** This operation is the inverse of erosion. The output pixel is set to the *maximum* value of the pixels in the input image's neighborhood. Dilation expands bright regions.

When an erosion is followed by a dilation (i.e., an opening), the effect is to remove all bright features that are smaller than the structuring element. If one chooses a structuring element (e.g., a disk with a radius of 50-100 pixels) that is significantly larger than any of the textural features of the nanofiber mat, the opening operation will effectively erase the mat from the image, leaving behind only the large, slowly varying background illumination. This generated background image can then be subtracted from the original image to yield the corrected result.

The primary advantage of this method is that it is fully automatic and does not require the user to manually select background regions. However, its performance is sensitive to the size and shape of the structuring element, and improper selection can lead to artifacts or incomplete background removal.

### **3.2 Frequency-Domain Correction via Homomorphic Filtering**

Homomorphic filtering is the most sophisticated of the techniques discussed, operating in the frequency domain to separate the illumination and reflectance components of an image. It is directly derived from the illumination-reflectance model, I(x,y)=L(x,y)×R(x,y), and leverages the assumption that illumination (L) is a low-frequency signal (it varies slowly across the image) while reflectance (R), which includes the mat's texture and edges, is a high-frequency signal (it varies rapidly).1

#### **3.2.1 Procedure**

The homomorphic filtering process involves several distinct mathematical steps 1:

1. Logarithmic Transform: The first step is to convert the multiplicative relationship into an additive one by taking the natural logarithm of the image. To avoid taking the log of zero, a small constant (e.g., 1\) is typically added to the image first:

   ln(I(x,y)+1)=ln(L(x,y)⋅R(x,y)+1)≈ln(L(x,y))+ln(R(x,y))  
2. **Fourier Transform:** A 2D Fast Fourier Transform (FFT) is applied to the log-transformed image. This converts the image from the spatial domain (pixels) to the frequency domain, where the image is represented by the magnitude and phase of various spatial frequencies. In this domain, the low-frequency illumination components are concentrated near the center of the FFT spectrum, while the high-frequency reflectance components are located further out.  
3. **High-Pass Filtering:** A high-pass filter is constructed and applied in the frequency domain. Common choices include Gaussian or Butterworth high-pass filters. This filter is designed to attenuate or completely remove the low-frequency components associated with ln(L) while preserving the high-frequency components associated with ln(R). The filter's cutoff frequency is a critical parameter that determines the boundary between what is considered "illumination" and what is considered "reflectance."  
4. **Inverse Fourier Transform:** An inverse FFT is applied to the filtered spectrum, transforming the image back into the spatial domain. The result is an image that primarily contains the high-frequency log-reflectance component.  
5. **Exponential Transform:** Finally, the exponential function (the inverse of the logarithm) is applied to the image to convert it back from the log domain, yielding the final corrected image, which is an estimate of the reflectance component, R(x,y).

Homomorphic filtering is an exceptionally powerful technique for simultaneously correcting non-uniform illumination and enhancing local contrast. However, it is the most computationally complex method and requires careful tuning of the filter parameters (e.g., filter type, cutoff frequency, order) to achieve optimal results.

### **Table 1: Comparative Analysis of Background Correction Methods**

To facilitate the selection of the most appropriate correction method, the following table summarizes the principles, advantages, disadvantages, and ideal use cases for each technique discussed.

| Method | Principle | Pros | Cons | Best For |
| :---- | :---- | :---- | :---- | :---- |
| **Flat-Field Correction** | Direct experimental measurement of the illumination field. | Highest physical accuracy; corrects for both illumination and fixed sensor defects. | Requires acquisition of additional calibration images; demands strict experimental control. | Rigorous quantitative studies where the highest accuracy is paramount. |
| **2D Polynomial Fitting** | Mathematical modeling of the background as a smooth surface. | Uses only the sample image; computationally efficient and robust for smooth gradients. | Requires sufficient visible background for sampling; assumes background is well-modeled by a low-order polynomial. | General-purpose correction for images with visible, smooth background gradients. |
| **Morphological Estimation** | Separation of background and foreground based on feature size. | Fully automatic; no user-defined background points needed. | Can introduce artifacts if structuring element is poorly chosen; may be less accurate than fitting methods. | Images where the mat covers most of the frame, leaving little background to sample for other methods. |
| **Homomorphic Filtering** | Separation of background and foreground based on spatial frequency. | Powerful and elegant separation of illumination and reflectance; enhances local contrast. | Computationally complex; requires careful tuning of filter parameters (cutoff, order). | Challenging cases with complex textures and illumination; applications where contrast enhancement is also desired. |

## **Section 4: Robust Segmentation of the Nanofiber Mat Region of Interest (ROI)**

After correcting for illumination artifacts, the next critical step is to precisely isolate the nanofiber mat from the rest of the image. This process, known as segmentation, creates a binary mask that defines the Region of Interest (ROI). This mask is essential, as it ensures that all subsequent thickness and uniformity calculations are performed exclusively on the pixels corresponding to the mat, excluding the hydrogel border and the polymer background. The specific design of the samples, featuring a dark hydrogel perimeter, provides a significant advantage that enables a highly robust and accurate segmentation strategy.

### **4.1 The Advantage of the Hydrogel Border: A High-Contrast Feature**

A common challenge in image segmentation is dealing with low-contrast boundaries or situations where a global intensity threshold fails due to residual illumination gradients.26 While adaptive thresholding methods can address this, they can be computationally slow and sensitive to parameter choices.28

The samples under analysis, however, possess a key feature that circumvents these difficulties: the dark conductive hydrogel that frames the mat. As seen in the provided images, this hydrogel creates two distinct, high-contrast interfaces:

1. An inner boundary between the bright white mat and the dark black hydrogel.  
2. An outer boundary between the dark black hydrogel and the medium-gray polymer substrate.

This sharp, predictable drop in intensity at the mat's edge is an ideal feature for edge detection algorithms. Instead of attempting to segment the mat based on its own internal (and potentially variable) intensity, a more reliable strategy is to detect the well-defined boundary created by the hydrogel. This approach makes the segmentation process fundamentally more robust and less susceptible to errors from incomplete background correction or inherent variations in mat brightness.

### **4.2 High-Fidelity Edge Detection with the Canny Algorithm**

To locate the high-contrast boundary of the mat, the **Canny edge detector** is the algorithm of choice. It is a multi-stage algorithm widely regarded for its superior performance in detecting true edges while being resilient to noise.29 It significantly outperforms simpler methods like Sobel or Laplacian filters because it incorporates intelligent processing steps to refine the edge map.

The stages of the Canny algorithm are as follows 29:

1. **Noise Reduction:** The algorithm first applies a Gaussian filter to the input image (the corrected grayscale image from the previous steps). This smoothing step is crucial as edge detection is inherently sensitive to high-frequency noise, which can lead to the detection of false edges. A 5x5 kernel is a common choice.  
2. **Finding Intensity Gradient:** The smoothed image is then filtered with Sobel kernels in both the horizontal (Gx​) and vertical (Gy​) directions. These operators calculate the first derivative of the intensity, providing the gradient magnitude (G=Gx2​+Gy2​​) and direction (θ=arctan(Gy​/Gx​)) for each pixel. The gradient magnitude is high at edges, and the gradient direction is perpendicular to the edge.  
3. **Non-Maximum Suppression:** This step thins the thick, blurry edges detected by the Sobel filter down to a single-pixel width. For each pixel, the algorithm checks if its gradient magnitude is a local maximum in the direction of its gradient. If a pixel is not a local maximum, it is suppressed (set to zero). This ensures that the final edges are sharp and well-defined.  
4. **Hysteresis Thresholding:** This is the final and most intelligent stage of the Canny algorithm. Instead of using a single threshold, it uses two: a high threshold (maxVal) and a low threshold (minVal).  
   * Any pixel with a gradient magnitude greater than maxVal is immediately classified as a "sure-edge."  
   * Any pixel with a gradient magnitude less than minVal is immediately discarded as a non-edge.  
   * Any pixel with a gradient magnitude between minVal and maxVal is classified as a "weak-edge." These weak-edge pixels are only kept in the final edge map if they are connected to a "sure-edge" pixel.  
     This process allows the algorithm to trace along fainter sections of an edge while simultaneously ignoring isolated noise pixels that happen to fall within the threshold range.

For this application, the Canny algorithm should be applied to the background-corrected grayscale image. The minVal and maxVal parameters will need to be tuned to reliably detect the strong edge of the hydrogel border while ignoring any minor texture within the mat itself.

### **4.3 Contour Extraction and ROI Mask Generation**

The output of the Canny algorithm is a binary image where white pixels represent the detected edges. The final step in segmentation is to convert this edge map into a solid mask representing the mat's area.

#### **4.3.1 Procedure**

1. **Find Contours:** A contour-finding algorithm, such as cv.findContours() available in the OpenCV library, is applied to the Canny edge map.31 This function traces the paths of connected white pixels and returns them as a list of contours, where each contour is a list of its constituent  
   (x,y) coordinates.  
2. **Select the Mat Contour:** The findContours function will likely detect multiple contours, including the outer boundary of the hydrogel and potentially some noise. The contour corresponding to the inner boundary of the hydrogel (the true mat boundary) needs to be selected. This can typically be done robustly by filtering the contours based on properties like area or position. For instance, the desired contour will likely be the largest or second-largest contour located near the center of the image.  
3. **Create Binary Mask:** Once the correct contour has been identified, a new, empty (all black) image of the same dimensions as the original is created. The selected contour is then drawn onto this new image using a solid white fill.

The resulting image is the final **binary ROI mask**. In this mask, all pixels corresponding to the nanofiber mat have a value of 1 (or 255 for an 8-bit image), and all other pixels have a value of 0\. This mask is a critical digital tool that will be used in all subsequent steps to ensure that calculations are confined precisely to the region of interest.

## **Section 5: Establishing the Intensity-to-Thickness Model via Empirical Calibration**

This section details the most critical phase of the analysis pipeline: building a quantitative bridge between the processed image data (a map of pixel intensities) and the desired physical property (a map of mat thickness). As established in Section 1, a purely theoretical model is impractical. Therefore, an empirical calibration is required. This involves creating a set of reference samples with known thicknesses, measuring their corresponding image intensities, and fitting a mathematical model to these data points. The resulting calibrated function, Thickness \= f(Intensity), becomes the predictive engine for the entire analysis.

### **5.1 Ground-Truth Metrology for Calibration Standards**

To build a reliable model, one must first obtain "ground-truth" thickness measurements for a series of calibration samples. These measurements must be performed using an independent, high-accuracy metrology technique. The choice of technique involves a trade-off between ultimate accuracy, cost, and whether the measurement is destructive.

#### **5.1.1 Method 1 (Gold Standard): Cross-Sectional Scanning Electron Microscopy (SEM)**

SEM provides a direct, high-resolution visualization of the mat's cross-section, making it the gold standard for thickness validation.33

* **Procedure:**  
  1. **Sample Preparation:** A representative piece of the nanofiber mat is carefully cut. To obtain a clean cross-section without compressing the porous structure, the sample is often cryo-fractured by freezing it in liquid nitrogen and then snapping it.  
  2. **Mounting and Coating:** The fractured sample is mounted on an SEM stub, oriented so that the cross-section is facing upward and is perpendicular to the electron beam. Since the polymer mat is non-conductive, it must be sputter-coated with a thin layer of a conductive material (e.g., gold, platinum, or carbon) to prevent charge buildup under the electron beam.34  
  3. **Imaging and Measurement:** The sample is imaged in the SEM. The resulting micrograph will show a clear view of the mat layer on top of the substrate. Using the scale bar provided on the SEM image and image analysis software (like ImageJ), the thickness of the mat can be measured directly. To account for local variations, it is best practice to take measurements at multiple locations along the cross-section and average them.36  
* **Advantages and Disadvantages:** SEM offers unparalleled accuracy and provides direct visual confirmation of the mat's internal structure and thickness.35 However, the technique is destructive, requires specialized equipment, and involves a time-consuming sample preparation process.

#### **5.1.2 Method 2 (Non-Destructive Alternative): Non-Contact Optical Profilometry**

Optical profilometry is a non-destructive technique that can measure surface topography with high precision, making it a viable alternative to SEM.38

* **Procedure:** The technique measures the "step height" from the bare substrate to the top surface of the nanofiber mat. A sample is prepared with a sharp edge (e.g., by masking part of the substrate during electrospinning or carefully cutting away a section of the mat). A white-light or laser-based profilometer scans across this edge. The instrument uses interferometry or laser triangulation to precisely measure the vertical difference between the substrate plane and the mat surface plane, yielding the thickness.39  
* **Advantages and Disadvantages:** The primary advantage of profilometry is that it is non-destructive and fast. However, it can be less accurate for highly porous and soft materials like nanofiber mats. The laser or light may partially penetrate the top layers of the mat before reflecting, and the definition of the "top surface" can be ambiguous, potentially leading to an underestimation of the true bulk thickness.38 It is crucial to use a non-contact method, as a contact-based profilometer or a simple micrometer screw gauge would compress the delicate mat structure, leading to significant measurement errors.38

### **5.2 Constructing the Calibration Curve: The Core Experiment**

With a ground-truth metrology method selected, the core calibration experiment can be performed. This involves systematically relating known physical thicknesses to measured image intensities.

#### **5.2.1 Procedure**

1. **Create Calibration Standards:** A series of at least 5-7 nanofiber mat samples must be produced. The most straightforward way to create samples of varying thickness is to hold all electrospinning process parameters (voltage, flow rate, solution concentration, collector distance) constant and systematically vary only the **deposition time**. This should yield a set of mats with a range of thicknesses.  
2. **Measure True Thickness:** For each mat in the calibration set, measure its average physical thickness using the chosen ground-truth method (SEM or profilometry). This value is the "ground truth" for that standard.  
3. **Image the Standards:** Acquire a digital image of each calibration standard. It is absolutely critical that every image is taken using the *exact same* optimized and corrected imaging protocol established in Sections 2 and 4\. The lighting, camera settings, geometry, and background correction procedure must be identical for all standards to ensure that intensity values are comparable.  
4. **Calculate Mean Intensity:** For each standard's corrected image, apply the segmentation mask to isolate the mat ROI. Then, calculate the mean grayscale intensity of all pixels within that ROI. This single value represents the average "whiteness" of that standard.  
5. **Plot the Data:** Create a 2D scatter plot. The independent variable, on the X-axis, is the "Mean Corrected Intensity (arbitrary units)." The dependent variable, on the Y-axis, is the "True Thickness (µm)" measured via the ground-truth method. Each calibration standard will be a single point on this plot.

### **5.3 Mathematical Modeling and Validation**

The final step is to fit a mathematical function to the plotted calibration data. This function will serve as the predictive model that converts any future intensity measurement into a thickness value.

* **Model Selection:** The choice of function should be guided by both the empirical data and the underlying physics.  
  * **Linear Fit:** A simple linear regression (y=mx+c) is the easiest to fit but is unlikely to be accurate over a wide range of thicknesses. It fails to account for the saturation effect predicted by Kubelka-Munk theory, where the mat's brightness stops increasing significantly after a certain thickness.  
  * **Polynomial Fit:** A second-order polynomial (y=ax2+bx+c) offers more flexibility and can capture some of the non-linearity.  
  * Exponential/Logarithmic Fit: Based on the physics of light transport in scattering media, a function that exhibits saturation is the most theoretically sound choice. An exponential rise to a maximum or a logarithmic function often provides an excellent fit. For example, a model derived from the principles of K-M theory might take a form like:

    Thickness=−A⋅ln(1−BIntensity​)

    where A and B are constants determined by the fitting process. B would represent the saturation intensity.  
* **Validation:** The quality of the chosen model is assessed by its goodness-of-fit to the calibration data. The **coefficient of determination (R²)** is the standard metric for this. An R² value close to 1.0 (e.g., \> 0.95) indicates that the model explains a large proportion of the variance in the data and is therefore a reliable predictor.41 The function that provides the best fit (highest R²) should be selected as the final  
  **calibration model**.

### **Table 2: Comparison of Ground-Truth Metrology Techniques**

| Technique | Principle | Pros | Cons | Resolution | Output |
| :---- | :---- | :---- | :---- | :---- | :---- |
| **Cross-Sectional SEM** | Electron beam imaging of a cryo-fractured sample. | Highest accuracy; direct visualization of internal structure; considered the 'gold standard'. | Destructive; requires extensive sample preparation and vacuum environment; time-consuming. | \~nm | 2D image of the mat's cross-section. |
| **Optical Profilometry** | Non-contact measurement of step height using interferometry or laser scanning. | Non-destructive; fast measurement; no vacuum required. | May be less accurate on soft/porous surfaces due to light penetration; measures surface topography only. | \~10s of nm to µm | 3D surface map or 2D height profile. |

## **Section 6: Quantitative Analysis and Visualization of Mat Uniformity**

With a fully calibrated and validated system, the final stage is to apply these tools to analyze the thickness uniformity of new, unknown samples. This section outlines the process of generating a quantitative thickness map from a sample image and then using statistical and visual methods to provide a comprehensive assessment of its uniformity. This final output delivers on the primary goal of moving beyond qualitative observation to a rigorous, data-driven analysis.

### **6.1 Generating the Quantitative Thickness Map**

The culmination of the previous steps is the ability to transform a standard digital image into a spatially resolved map of physical thickness.

#### **6.1.1 Procedure**

For any new sample image to be analyzed, the following steps are performed:

1. **Acquire and Correct Image:** The image must be acquired and corrected using the identical, standardized protocol developed in Sections 2 and 4\. This includes consistent lighting, camera settings, geometry, background correction, and segmentation to generate the corrected grayscale intensity image and the corresponding ROI mask.  
2. **Apply Calibration Model:** The calibrated function, Thickness \= f(Intensity), established in Section 5.3 is applied to the image. This operation is performed on a pixel-by-pixel basis for every pixel located *within* the ROI defined by the segmentation mask.  
3. **Generate Thickness Map:** The output of this operation is a new 2D array, or image, of the same dimensions as the original. However, the value of each pixel in this new array is no longer an arbitrary intensity unit but a calculated physical thickness, typically in micrometers (µm).

This final 2D array is the **quantitative thickness map**. It is the foundational dataset from which all subsequent uniformity metrics and visualizations are derived.

### **6.2 Statistical Metrics for Uniformity: The Quantitative Scorecard**

While a visual map is insightful, objective comparison between different samples requires quantitative metrics. These statistics distill the complex spatial data of the thickness map into a few key numbers that describe the mat's overall uniformity. The calculations are performed on the array of thickness values from all pixels within the ROI.

* Mean Thickness (μ): The average thickness across the entire mat. This provides the central tendency of the thickness distribution.

  μ=N∑i=1N​Ti​​

  where Ti​ is the thickness of the i-th pixel and N is the total number of pixels in the ROI.  
* Standard Deviation (σ): This metric quantifies the absolute dispersion or spread of thickness values around the mean. A larger standard deviation indicates greater variability and less uniformity. Its units are the same as the mean (e.g., µm).

  σ=N∑i=1N​(Ti​−μ)2​​  
* **Coefficient of Variation (CV):** This is the **most important and powerful metric for assessing uniformity**.42 It is defined as the ratio of the standard deviation to the mean:  
  CV=μσ​

  The CV, often expressed as a percentage, is a dimensionless number that represents the relative standard deviation. Its key advantage is that it normalizes the variability of the sample with respect to its average thickness. This makes it ideal for comparing the uniformity of different mats, even if they have widely different average thicknesses. For example, a standard deviation of 5 µm might be considered highly non-uniform for a 20 µm mat (CV \= 0.25 or 25%) but quite uniform for a 200 µm mat (CV \= 0.025 or 2.5%). A lower CV always indicates a more uniform mat.42

### **6.3 Spatial Visualization for Deeper Insight: Where is the Non-Uniformity?**

Statistical metrics provide a summary of *how* non-uniform a mat is, but they do not reveal *where* the non-uniformities are located. Spatial visualizations are crucial for understanding the nature of the thickness variations and can provide valuable, actionable feedback for optimizing the electrospinning process.

* **2D Color Heatmap:** This is the most direct and intuitive visualization of the quantitative thickness map. A pseudocolor map (e.g., 'viridis', 'jet', or 'hot') is applied to the map, where color is scaled to correspond to thickness. This allows for the immediate identification of thicker (hot colors) and thinner (cool colors) regions, revealing patterns, gradients, or random fluctuations.  
* **3D Surface Plot:** The thickness map can be treated as a height map and rendered as a 3D surface plot. The x and y axes represent the spatial dimensions of the mat, and the z-axis represents the thickness. This type of plot can be very effective at exaggerating subtle variations and making large-scale gradients or waviness more apparent.  
* **Line Profile Analysis:** This technique involves extracting and plotting the thickness values along one or more lines drawn across the mat.17 This is an excellent method for diagnosing systematic non-uniformities.  
  * **Procedure:** Using software tools (e.g., the Plot Profile function in ImageJ 46), one can draw lines horizontally, vertically, and diagonally across the thickness map. The software then generates a 2D plot with distance along the line on the x-axis and thickness on the y-axis.  
  * **Interpretation:** A flat line profile indicates perfect uniformity along that cross-section. A profile that is convex (higher in the middle) suggests the mat is thicker in the center, a common characteristic of basic electrospinning setups. A tilted profile indicates a gradient in thickness across the mat. Analyzing multiple line profiles provides a powerful diagnostic tool for understanding the spatial distribution of material.

### **Table 3: Summary of Uniformity Metrics**

| Metric | Formula | Interpretation |
| :---- | :---- | :---- |
| **Mean Thickness (μ)** | N∑Ti​​ | The average thickness of the mat, providing a measure of the central value of the distribution. Units are in µm. |
| **Standard Deviation (σ)** | N∑(Ti​−μ)2​​ | The absolute spread or variability of thickness values around the mean. A direct measure of non-uniformity in physical units (µm). |
| **Coefficient of Variation (CV)** | μσ​ | The relative standard deviation. A dimensionless, normalized measure of uniformity. **Lower values indicate higher uniformity.** It is the best metric for comparing the uniformity of samples with different average thicknesses. |

## **Conclusions and Recommendations**

This report has detailed a comprehensive, first-principles methodology for the non-destructive analysis of electrospun mat thickness and uniformity using digital image processing. By systematically addressing the challenges of non-uniform illumination, reflective substrates, and the complex optical properties of scattering media, this protocol provides a robust and scientifically defensible alternative to manual or destructive measurement techniques.

The key conclusions and recommendations derived from this analysis are as follows:

1. **An Empirical Calibration is Essential:** The optical behavior of the nanofiber mat is governed by complex, multiple-light-scattering phenomena best described by Kubelka-Munk theory. A direct theoretical calculation of thickness from reflectance is impractical. The most robust approach is to **empirically calibrate** the system by relating measured image intensity to ground-truth thickness values obtained from an independent, high-accuracy technique like cross-sectional SEM or non-contact profilometry.  
2. **Rigorous Image Acquisition is Non-Negotiable:** The accuracy of the entire method hinges on the quality and consistency of the acquired images. It is strongly recommended to implement a standardized imaging protocol that includes:  
   * Diffuse, stable illumination.  
   * **Cross-polarization** to minimize specular reflections from the substrate.  
   * Fixed manual camera settings (ISO, aperture, shutter speed, white balance).  
   * A fixed, normal imaging geometry.  
3. **Illumination Correction Must Be Performed:** The non-uniform illumination present in the imaging setup must be corrected.  
   * **Primary Recommendation:** **Flat-field correction** is the experimental gold standard and should be used for the highest quantitative accuracy.  
   * **Alternative Recommendation:** **2D polynomial surface fitting** is a powerful and effective computational alternative that is well-suited to the smooth background gradients observed and can be used if experimental flat-fielding is not feasible.  
4. **Leverage High-Contrast Features for Segmentation:** The dark hydrogel border is a significant advantage. A segmentation strategy based on the **Canny edge detector** followed by **contour finding** to identify this border will yield a more robust and accurate ROI mask than methods based on thresholding the mat itself.  
5. **The Coefficient of Variation (CV) is the Definitive Metric for Uniformity:** While mean thickness and standard deviation are important descriptors, the **CV (σ/μ)** should be adopted as the primary metric for quantifying and comparing the uniformity of different mats. As a normalized, dimensionless measure, it provides the most meaningful comparison between samples of varying average thicknesses.

By implementing this systematic pipeline—from establishing the foundational physics, to optimizing image acquisition and correction, to performing a rigorous empirical calibration, and finally to calculating quantitative metrics—it is possible to develop a reliable, non-destructive, and high-throughput method for characterizing the thickness uniformity of electrospun nanofiber mats. This approach not only solves the immediate analytical challenge but also provides a framework for process control and optimization, enabling a deeper understanding and improvement of the electrospinning fabrication process.

#### **Works cited**

1. Homomorphic filtering – part 1 » Steve on Image Processing with MATLAB, accessed August 1, 2025, [https://blogs.mathworks.com/steve/2013/06/25/homomorphic-filtering-part-1/](https://blogs.mathworks.com/steve/2013/06/25/homomorphic-filtering-part-1/)  
2. Measuring Physical Properties of Electrospun Nanofiber Mats for Different Biomedical Applications \- MDPI, accessed August 1, 2025, [https://www.mdpi.com/2077-0375/13/5/488](https://www.mdpi.com/2077-0375/13/5/488)  
3. Review on Electrospun Nanofiber-Applied Products \- MDPI, accessed August 1, 2025, [https://www.mdpi.com/2073-4360/13/13/2087](https://www.mdpi.com/2073-4360/13/13/2087)  
4. Optical Properties of Electrospun Nanofiber Mats \- MDPI, accessed August 1, 2025, [https://www.mdpi.com/2077-0375/13/4/441](https://www.mdpi.com/2077-0375/13/4/441)  
5. Optical Properties of Electrospun Nanofiber Mats \- ResearchGate, accessed August 1, 2025, [https://www.researchgate.net/publication/370100257\_Optical\_Properties\_of\_Electrospun\_Nanofiber\_Mats](https://www.researchgate.net/publication/370100257_Optical_Properties_of_Electrospun_Nanofiber_Mats)  
6. Optical Properties of Electrospun Nanofiber Mats \- PMC \- PubMed Central, accessed August 1, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC10146296/](https://pmc.ncbi.nlm.nih.gov/articles/PMC10146296/)  
7. Electrospun hybrid nanofibers: Fabrication, characterization, and biomedical applications \- Frontiers, accessed August 1, 2025, [https://www.frontiersin.org/journals/bioengineering-and-biotechnology/articles/10.3389/fbioe.2022.986975/full](https://www.frontiersin.org/journals/bioengineering-and-biotechnology/articles/10.3389/fbioe.2022.986975/full)  
8. (PDF) The modified Beer-Lambert law revisited \- ResearchGate, accessed August 1, 2025, [https://www.researchgate.net/publication/7294747\_The\_modified\_Beer-Lambert\_law\_revisited](https://www.researchgate.net/publication/7294747_The_modified_Beer-Lambert_law_revisited)  
9. Modified Beer-Lambert law for blood flow \- PMC, accessed August 1, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC4242038/](https://pmc.ncbi.nlm.nih.gov/articles/PMC4242038/)  
10. Modified Beer-Lambert Law \- openfnirs, accessed August 1, 2025, [https://openfnirs.org/2024/01/01/modified-beer-lambert-law/](https://openfnirs.org/2024/01/01/modified-beer-lambert-law/)  
11. Uniform-thickness electrospun nanofiber mat production system ..., accessed August 1, 2025, [https://pubmed.ncbi.nlm.nih.gov/33257811/](https://pubmed.ncbi.nlm.nih.gov/33257811/)  
12. Kubelka–Munk theory \- Wikipedia, accessed August 1, 2025, [https://en.wikipedia.org/wiki/Kubelka%E2%80%93Munk\_theory](https://en.wikipedia.org/wiki/Kubelka%E2%80%93Munk_theory)  
13. KUBELKA-MUNK THEORY IN DESCRIBING OPTICAL PROPERTIES OF PAPER (I) \- CORE, accessed August 1, 2025, [https://core.ac.uk/download/pdf/14434725.pdf](https://core.ac.uk/download/pdf/14434725.pdf)  
14. Applicability conditions of the Kubelka–Munk theory \- Optics Express, accessed August 1, 2025, [https://opg.optica.org/fulltext.cfm?uri=ao-36-22-5580](https://opg.optica.org/fulltext.cfm?uri=ao-36-22-5580)  
15. Kubelka-Munk theory – Knowledge and References \- Taylor & Francis, accessed August 1, 2025, [https://taylorandfrancis.com/knowledge/Engineering\_and\_technology/Electrical\_%26\_electronic\_engineering/Kubelka-Munk\_theory/](https://taylorandfrancis.com/knowledge/Engineering_and_technology/Electrical_%26_electronic_engineering/Kubelka-Munk_theory/)  
16. Three-Dimensional Surface Profile Intensity Correction for Spatially-Modulated Imaging, accessed August 1, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC2756969/](https://pmc.ncbi.nlm.nih.gov/articles/PMC2756969/)  
17. Method for assessing coating uniformity of angioplasty balloons coated with poly(lactic-co-glycolic acid) nanoparticles loaded w \- bioRxiv, accessed August 1, 2025, [https://www.biorxiv.org/content/10.1101/2023.01.18.524614v1.full.pdf](https://www.biorxiv.org/content/10.1101/2023.01.18.524614v1.full.pdf)  
18. Background illumination correction – Novel context-based segmentation algorithms for intelligent microscopy \- Birmingham Blogs, accessed August 1, 2025, [https://blog.bham.ac.uk/intellimic/g-landini-software/background-illumination-correction/](https://blog.bham.ac.uk/intellimic/g-landini-software/background-illumination-correction/)  
19. Image Vignetting Correction Using a Deformable Radial Polynomial Model \- MDPI, accessed August 1, 2025, [https://www.mdpi.com/1424-8220/23/3/1157](https://www.mdpi.com/1424-8220/23/3/1157)  
20. Enhancement of Curve-Fitting Image Compression Using Hyperbolic Function \- MDPI, accessed August 1, 2025, [https://www.mdpi.com/2073-8994/11/2/291](https://www.mdpi.com/2073-8994/11/2/291)  
21. A Smooth Non-Iterative Local Polynomial (SNILP) Model of Image Vignetting \- PMC, accessed August 1, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC8588465/](https://pmc.ncbi.nlm.nih.gov/articles/PMC8588465/)  
22. Morphological Filtering — skimage 0.25.2 documentation, accessed August 1, 2025, [https://scikit-image.org/docs/0.25.x/auto\_examples/applications/plot\_morphology.html](https://scikit-image.org/docs/0.25.x/auto_examples/applications/plot_morphology.html)  
23. Different Morphological Operations in Image Processing \- GeeksforGeeks, accessed August 1, 2025, [https://www.geeksforgeeks.org/computer-vision/different-morphological-operations-in-image-processing/](https://www.geeksforgeeks.org/computer-vision/different-morphological-operations-in-image-processing/)  
24. OpenCV : homomorphic filter \- c++ \- Stack Overflow, accessed August 1, 2025, [https://stackoverflow.com/questions/23740934/opencv-homomorphic-filter](https://stackoverflow.com/questions/23740934/opencv-homomorphic-filter)  
25. homomorphic filter \- python overflow \- Signal Processing Stack Exchange, accessed August 1, 2025, [https://dsp.stackexchange.com/questions/42476/homomorphic-filter-python-overflow](https://dsp.stackexchange.com/questions/42476/homomorphic-filter-python-overflow)  
26. Thresholding — skimage 0.25.2 documentation, accessed August 1, 2025, [https://scikit-image.org/docs/0.25.x/auto\_examples/applications/plot\_thresholding\_guide.html](https://scikit-image.org/docs/0.25.x/auto_examples/applications/plot_thresholding_guide.html)  
27. Thresholding (image processing) \- Wikipedia, accessed August 1, 2025, [https://en.wikipedia.org/wiki/Thresholding\_(image\_processing)](https://en.wikipedia.org/wiki/Thresholding_\(image_processing\))  
28. Image Thresholding \- OpenCV Documentation, accessed August 1, 2025, [https://docs.opencv.org/4.x/d7/d4d/tutorial\_py\_thresholding.html](https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html)  
29. Canny Edge Detection \- OpenCV Documentation, accessed August 1, 2025, [https://docs.opencv.org/4.x/da/d22/tutorial\_py\_canny.html](https://docs.opencv.org/4.x/da/d22/tutorial_py_canny.html)  
30. Implement Canny Edge Detector in Python using OpenCV \- GeeksforGeeks, accessed August 1, 2025, [https://www.geeksforgeeks.org/machine-learning/implement-canny-edge-detector-in-python-using-opencv/](https://www.geeksforgeeks.org/machine-learning/implement-canny-edge-detector-in-python-using-opencv/)  
31. OpenCV Detecting Contours Project \- DataFlair, accessed August 1, 2025, [https://data-flair.training/blogs/opencv-detecting-contours-project/](https://data-flair.training/blogs/opencv-detecting-contours-project/)  
32. Contours : Getting Started \- OpenCV Documentation, accessed August 1, 2025, [https://docs.opencv.org/4.x/d4/d73/tutorial\_py\_contours\_begin.html](https://docs.opencv.org/4.x/d4/d73/tutorial_py_contours_begin.html)  
33. (PDF) Non‐destructive thickness measurement of optically ..., accessed August 1, 2025, [https://www.researchgate.net/publication/376811428\_Non-destructive\_thickness\_measurement\_of\_optically\_scattering\_polymer\_films\_using\_image\_processing](https://www.researchgate.net/publication/376811428_Non-destructive_thickness_measurement_of_optically_scattering_polymer_films_using_image_processing)  
34. Boehmite Nanofiber–Polymethylsilsesquioxane Composite Macroporous Monoliths with High Diffuse Reflectance and Micromachinability | ACS Applied Polymer Materials, accessed August 1, 2025, [https://pubs.acs.org/doi/full/10.1021/acsapm.5c01749](https://pubs.acs.org/doi/full/10.1021/acsapm.5c01749)  
35. How Do You Measure Thin Film Sem Thickness? A Step-By-Step Guide To High-Resolution Analysis \- Kintek Solution, accessed August 1, 2025, [https://kindle-tech.com/faqs/how-do-you-measure-thin-film-sem-thickness](https://kindle-tech.com/faqs/how-do-you-measure-thin-film-sem-thickness)  
36. SEM Analysis of Hollow Nanofibers \- YouTube, accessed August 1, 2025, [https://www.youtube.com/watch?v=DRD4RdmTwVE](https://www.youtube.com/watch?v=DRD4RdmTwVE)  
37. SEM cross-sectional analysis of thickness determination for the prepared thin films \- ResearchGate, accessed August 1, 2025, [https://www.researchgate.net/figure/SEM-cross-sectional-analysis-of-thickness-determination-for-the-prepared-thin-films\_fig1\_342875134](https://www.researchgate.net/figure/SEM-cross-sectional-analysis-of-thickness-determination-for-the-prepared-thin-films_fig1_342875134)  
38. Thickness measurement for Electrospun membrane \- ElectrospinTech, accessed August 1, 2025, [http://electrospintech.com/thickness-measurement.html](http://electrospintech.com/thickness-measurement.html)  
39. A non-destructive method for thickness measurement of thin electrospun membranes using white light profilometry | Request PDF \- ResearchGate, accessed August 1, 2025, [https://www.researchgate.net/publication/227195269\_A\_non-destructive\_method\_for\_thickness\_measurement\_of\_thin\_electrospun\_membranes\_using\_white\_light\_profilometry](https://www.researchgate.net/publication/227195269_A_non-destructive_method_for_thickness_measurement_of_thin_electrospun_membranes_using_white_light_profilometry)  
40. (PDF) Surface Roughness of Electrospun Nanofibrous Mats by a Novel Image Processing Technique \- ResearchGate, accessed August 1, 2025, [https://www.researchgate.net/publication/321725579\_Surface\_Roughness\_of\_Electrospun\_Nanofibrous\_Mats\_by\_a\_Novel\_Image\_Processing\_Technique](https://www.researchgate.net/publication/321725579_Surface_Roughness_of_Electrospun_Nanofibrous_Mats_by_a_Novel_Image_Processing_Technique)  
41. Estimation of Digital Porosity of Electrospun Veils by Image Analysis \- PMC, accessed August 1, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC10820155/](https://pmc.ncbi.nlm.nih.gov/articles/PMC10820155/)  
42. Coefficient of variation \- Wikipedia, accessed August 1, 2025, [https://en.wikipedia.org/wiki/Coefficient\_of\_variation](https://en.wikipedia.org/wiki/Coefficient_of_variation)  
43. Design and simulation of Multilayer coatings for a multi-channel Wolter-like x-ray imager with large field of view and high resolution \- OSTI, accessed August 1, 2025, [https://www.osti.gov/servlets/purl/2212854](https://www.osti.gov/servlets/purl/2212854)  
44. Method for assessing coating uniformity of angioplasty balloons, accessed August 1, 2025, [https://tandf.figshare.com/articles/journal\_contribution/Method\_for\_assessing\_coating\_uniformity\_of\_angioplasty\_balloons\_coated\_with\_poly\_lactic-co-glycolic\_acid\_nanoparticles\_loaded\_with\_quercetin/24451818](https://tandf.figshare.com/articles/journal_contribution/Method_for_assessing_coating_uniformity_of_angioplasty_balloons_coated_with_poly_lactic-co-glycolic_acid_nanoparticles_loaded_with_quercetin/24451818)  
45. Base Sheet Structures that Control Coating Uniformity: Effects of Length Scale, accessed August 1, 2025, [https://www.researchgate.net/publication/267781929\_Base\_Sheet\_Structures\_that\_Control\_Coating\_Uniformity\_Effects\_of\_Length\_Scale](https://www.researchgate.net/publication/267781929_Base_Sheet_Structures_that_Control_Coating_Uniformity_Effects_of_Length_Scale)  
46. GetProfileExample.txt, accessed August 1, 2025, [https://imagej.net/ij/macros/GetProfileExample.txt](https://imagej.net/ij/macros/GetProfileExample.txt)  
47. Plot Profile \- ImageJ Documentation Wiki, accessed August 1, 2025, [https://imagejdocu.list.lu/gui/analyze/plot\_profile](https://imagejdocu.list.lu/gui/analyze/plot_profile)