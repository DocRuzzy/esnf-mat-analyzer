# Multiscale Uniformity Analysis Guide

## Overview

Multiscale uniformity analysis is an advanced technique for evaluating the consistency of nanofiber mat thickness across multiple spatial scales. Unlike traditional uniformity metrics that provide a single global score, multiscale analysis reveals how uniformity varies from fine-scale texture variations to large-scale thickness gradients.

## What is Multiscale Analysis?

### The Concept

Real-world nanofiber mats have structure and variations at multiple scales:

1. **Fine Scale (Scale 1-2)**: Individual fiber texture, local density variations
2. **Medium Scale (Scale 3)**: Small defects, fiber bundle variations, local thickness patches
3. **Large Scale (Scale 0)**: Overall thickness gradients, edge effects, processing variations

Traditional analysis methods only capture one aspect of uniformity, missing the multi-scale nature of these materials.

### Why It Matters

Different applications require different types of uniformity:
- **Filtration**: Fine-scale uniformity is critical for consistent pore structure
- **Structural composites**: Medium-scale uniformity affects mechanical properties
- **Barrier applications**: Large-scale uniformity ensures consistent coverage

## Wavelets: The Mathematical Foundation

### What are Wavelets?

Wavelets are mathematical functions that decompose a signal (in our case, a thickness map) into components at different scales and frequencies. Think of them as "mathematical microscopes" that can zoom in and out to examine different levels of detail.

#### Key Properties:
- **Localized**: Unlike Fourier transforms, wavelets are localized in both space and frequency
- **Multi-resolution**: They naturally separate information at different scales
- **Efficient**: Fast algorithms make them practical for image analysis

### Wavelet Decomposition Process

When we apply wavelet decomposition to a thickness map:

```
Original Thickness Map (256×256)
                ↓
        Wavelet Decomposition
                ↓
┌─────────────────────────────────────┐
│ Scale 0: Approximation (128×128)    │  ← Large-scale trends
│ Scale 1: Details (128×128)          │  ← Medium-high frequency
│ Scale 2: Details (64×64)            │  ← Medium frequency  
│ Scale 3: Details (32×32)            │  ← Medium-low frequency
│ Scale 4: Details (16×16)            │  ← Fine details
└─────────────────────────────────────┘
```

#### Scale Interpretation:
- **Scale 0 (Approximation)**: Overall thickness distribution, large-scale gradients
- **Scale 1**: High-frequency details, individual fiber variations
- **Scale 2**: Medium-high frequency, small fiber bundles
- **Scale 3**: Medium frequency, processing defects, thickness patches
- **Scale 4**: Fine frequency, surface texture

### Detail Coefficients

Each detail scale contains three types of coefficients:
- **Horizontal (cH)**: Captures vertical edges and horizontal variations
- **Vertical (cV)**: Captures horizontal edges and vertical variations  
- **Diagonal (cD)**: Captures diagonal features and corner-like structures

## What the Analysis Evaluates

### Scale-Specific Uniformity Metrics

#### Scale 0 (Approximation Coefficients)
- **What it measures**: Large-scale thickness uniformity
- **Metric used**: Coefficient of Variation (CV)
- **Physical meaning**: How consistent is the overall thickness across the mat?
- **Typical values**: 
  - 0.9-1.0: Excellent large-scale uniformity
  - 0.7-0.9: Good uniformity with minor gradients
  - 0.5-0.7: Moderate uniformity with noticeable thickness variations
  - <0.5: Poor uniformity with significant thickness gradients

#### Scales 1-4 (Detail Coefficients)
- **What it measures**: Texture uniformity at different spatial frequencies
- **Metric used**: Normalized Standard Deviation (std/range)
- **Physical meaning**: How consistent are the local thickness variations?
- **Why not CV**: Detail coefficients often have near-zero means, making CV unstable

### Mathematical Details

#### For Scale 0 (Approximation):
```
CV = σ / |μ|
Uniformity = exp(-CV)
```
Where:
- σ = standard deviation of approximation coefficients
- μ = mean of approximation coefficients
- Exponential decay converts CV to 0-1 uniformity score

#### For Scales 1-4 (Details):
```
Normalized_Std = σ / (max - min)
Uniformity = 1 / (1 + Normalized_Std)
```
Where:
- σ = standard deviation of detail coefficients
- (max - min) = range of detail coefficients
- Inverse relationship converts to 0-1 uniformity score

## ROI Masking and Analysis Focus

### Why ROI Masking is Critical

The analysis focuses only on the Region of Interest (ROI) - typically the nanofiber mat area, excluding:
- Rulers or measurement scales
- Background regions
- Edge artifacts
- Non-mat areas

### How ROI Masking Works

1. **Pre-decomposition**: ROI mask is defined on the original image
2. **Post-decomposition masking**: ROI mask is downsampled to match each scale's resolution
3. **Coefficient extraction**: Only coefficients within the ROI are analyzed
4. **Scale-specific masking**: Each scale gets appropriately sized mask

```python
# Example for Scale 1 detail coefficients (128×128)
if original_mask.shape == (256, 256):
    # Downsample mask to match coefficient size
    downsampled_mask = original_mask[::2, ::2]  # Now (128×128)
    
    # Apply to each detail coefficient
    masked_horizontal = horizontal_coeffs[downsampled_mask]
    masked_vertical = vertical_coeffs[downsampled_mask]
    masked_diagonal = diagonal_coeffs[downsampled_mask]
```

## Interpretation Guide

### Individual Scale Scores

#### Scale 0 (Large-scale uniformity):
- **High scores (>0.8)**: Consistent overall thickness, minimal gradients
- **Medium scores (0.5-0.8)**: Some thickness variation, acceptable for most applications
- **Low scores (<0.5)**: Significant thickness gradients, may indicate processing issues

#### Scales 1-2 (Fine-scale uniformity):
- **High scores (>0.9)**: Very consistent fiber texture, excellent surface quality
- **Medium scores (0.7-0.9)**: Good texture consistency, minor local variations
- **Low scores (<0.7)**: Inconsistent texture, possible fiber density issues

#### Scales 3-4 (Medium-scale uniformity):
- **High scores (>0.8)**: Minimal defects, consistent local thickness
- **Medium scores (0.6-0.8)**: Some local variations, generally acceptable
- **Low scores (<0.6)**: Noticeable defects or thickness patches

### Overall Multiscale Score

The overall score is typically the average or weighted average of all scale scores:

```
Overall_Score = Σ(Scale_i_Uniformity) / Number_of_Scales
```

#### Overall Score Interpretation:
- **0.9-1.0**: Excellent uniformity across all scales
- **0.8-0.9**: Good uniformity, suitable for demanding applications
- **0.7-0.8**: Moderate uniformity, acceptable for many applications
- **0.6-0.7**: Fair uniformity, may need process optimization
- **<0.6**: Poor uniformity, significant process issues

## Practical Applications

### Quality Control

Use multiscale analysis to:
- **Identify processing issues**: Poor Scale 0 → temperature/speed problems
- **Detect equipment problems**: Poor Scale 1-2 → nozzle issues
- **Monitor consistency**: Track scores over time for process control

### Process Optimization

- **Fine-tune parameters**: Adjust based on which scales are problematic
- **Validate improvements**: Quantify the effect of process changes
- **Set quality standards**: Define acceptable ranges for each scale

### Research and Development

- **Material characterization**: Understand structure at multiple scales
- **Comparative studies**: Compare different materials or processes
- **Publication**: Provide quantitative, multi-scale characterization

## Technical Implementation

### Wavelet Choice

The analyzer uses Daubechies 4 (db4) wavelets by default:
- **Compact support**: Good localization in space
- **Orthogonal**: No information overlap between scales
- **Smooth**: Suitable for continuous thickness data

### Computational Considerations

- **Memory**: Each scale requires storage for coefficient arrays
- **Speed**: Fast wavelet transform is O(N) complexity
- **Precision**: Uses double-precision floating point for accuracy

### Validation and Robustness

- **NaN handling**: Invalid values are removed before analysis
- **Edge effects**: ROI masking minimizes boundary artifacts
- **Scale validation**: Ensures sufficient data points for meaningful statistics

## Common Issues and Solutions

### Problem: All scales return 0.0
**Cause**: Insufficient ROI coverage or corrupted data
**Solution**: Check ROI selection, verify data quality

### Problem: Scale 0 much lower than other scales
**Cause**: Large-scale thickness gradients
**Solution**: Check processing parameters, substrate leveling

### Problem: Scales 1-2 much lower than others
**Cause**: Fine-scale texture inconsistency
**Solution**: Check fiber formation parameters, nozzle condition

### Problem: Inconsistent results between runs
**Cause**: ROI selection variation, edge effects
**Solution**: Standardize ROI selection, use larger safety margins

## Best Practices

### ROI Selection
1. **Exclude rulers**: Ensure ROI covers only the mat area
2. **Avoid edges**: Leave margin from mat boundaries
3. **Consistent size**: Use similar ROI sizes for comparative studies
4. **Center on uniform areas**: Avoid obviously defective regions for baseline measurements

### Data Quality
1. **Sufficient resolution**: Ensure adequate pixels for analysis
2. **Proper calibration**: Verify thickness measurement accuracy
3. **Clean data**: Remove artifacts and invalid measurements
4. **Adequate coverage**: ROI should cover representative mat area

### Interpretation
1. **Consider application**: Weight scales based on end-use requirements
2. **Compare relatively**: Use similar samples as references
3. **Track trends**: Monitor changes over time or process conditions
4. **Validate with physical testing**: Correlate with mechanical or performance tests

## Future Enhancements

### Potential Improvements
- **Adaptive windowing**: Variable window sizes for spatial analysis
- **Wavelet selection**: Automatic choice based on data characteristics
- **Statistical modeling**: Confidence intervals and significance testing
- **Real-time analysis**: Integration with production monitoring systems

### Research Opportunities
- **Correlation studies**: Link multiscale scores with performance metrics
- **Machine learning**: Predictive models based on multiscale features
- **3D analysis**: Extension to volumetric uniformity assessment
- **Multi-modal**: Combine with other characterization techniques

## Conclusion

Multiscale uniformity analysis provides a comprehensive, quantitative assessment of nanofiber mat quality across multiple spatial scales. By leveraging wavelet decomposition, this technique reveals detailed information about material structure that traditional methods miss, enabling better quality control, process optimization, and research insights.

The key advantages are:
- **Multi-scale perspective**: Captures uniformity at all relevant spatial scales
- **Quantitative assessment**: Provides objective, reproducible metrics
- **ROI-focused**: Analyzes only the relevant material region
- **Research-grade**: Suitable for scientific publication and detailed analysis

Understanding these principles enables users to make informed decisions about material quality, process parameters, and application suitability based on comprehensive uniformity characterization.
