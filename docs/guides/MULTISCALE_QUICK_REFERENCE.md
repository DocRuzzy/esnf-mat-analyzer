# Multiscale Uniformity Quick Reference

## Scale Interpretation at a Glance

| Scale | What It Measures | Physical Meaning | Good Score | Typical Issues |
|-------|------------------|------------------|------------|----------------|
| **Scale 0** | Large-scale trends | Overall thickness gradients | >0.8 | Temperature variations, collector distance |
| **Scale 1** | Fine details | Individual fiber variations | >0.9 | Nozzle condition, solution properties |
| **Scale 2** | Medium-fine | Small fiber bundles | >0.9 | Local density variations |
| **Scale 3** | Medium | Processing defects, patches | >0.8 | Air flow, substrate irregularities |
| **Scale 4** | Coarse details | Surface texture | >0.8 | Equipment vibration, charge fluctuations |

## Score Interpretation

| Score Range | Quality Level | Recommendation |
|-------------|---------------|----------------|
| **0.9 - 1.0** | Excellent | Production ready |
| **0.8 - 0.9** | Good | Acceptable for most applications |
| **0.7 - 0.8** | Moderate | Consider optimization |
| **0.6 - 0.7** | Fair | Process tuning needed |
| **< 0.6** | Poor | Significant issues, review all parameters |

## Troubleshooting Guide

### Poor Scale 0 (Large-scale uniformity)
**Symptoms:** Overall score dominated by low Scale 0  
**Check:**
- Substrate temperature uniformity
- Collector distance consistency
- Solution flow rate stability
- Electric field uniformity

### Poor Scales 1-2 (Fine-scale uniformity)
**Symptoms:** High Scale 0, low Scales 1-2  
**Check:**
- Nozzle tip condition and cleanliness
- Solution concentration and viscosity
- Charge density and voltage stability
- Environmental humidity

### Poor Scales 3-4 (Medium-scale uniformity)
**Symptoms:** Good fine scales, poor medium scales  
**Check:**
- Air flow patterns and turbulence
- Substrate surface preparation
- Processing chamber cleanliness
- Equipment vibration levels

### Poor All Scales
**Symptoms:** Consistently low scores across all scales  
**Action:** Fundamental process review required
- Verify basic equipment function
- Check all environmental conditions
- Review solution preparation
- Validate measurement calibration

## ROI (Region of Interest) Guidelines

✅ **Include in ROI:**
- Representative mat area
- Central, uniform regions
- Adequate sample size

❌ **Exclude from ROI:**
- Ruler or measurement scales
- Edge effects near boundaries
- Obvious processing defects
- Background or non-mat areas

## Mathematical Background (Summary)

### Scale 0 (Approximation Coefficients)
```
Uniformity = exp(-CV) where CV = σ/|μ|
```
- Uses Coefficient of Variation because approximation has meaningful mean
- Exponential decay emphasizes low variation

### Scales 1-4 (Detail Coefficients)
```
Uniformity = 1/(1 + normalized_std) where normalized_std = σ/range
```
- Uses normalized standard deviation because details have near-zero means
- Range normalization makes scores comparable across scales

## Wavelet Decomposition Overview

```
Original Image (256×256)
        ↓ Wavelet Decomposition
┌─────────────────────────────┐
│ Scale 0: Approx (128×128)   │ ← Overall trends
│ Scale 1: Details (128×128)  │ ← High frequency
│ Scale 2: Details (64×64)    │ ← Medium-high frequency
│ Scale 3: Details (32×32)    │ ← Medium frequency
│ Scale 4: Details (16×16)    │ ← Low frequency details
└─────────────────────────────┘
```

## Best Practices

1. **Consistent ROI Selection:** Use similar size and position for comparative studies
2. **Adequate Resolution:** Ensure sufficient pixels for meaningful analysis
3. **Clean Data:** Remove artifacts and invalid measurements before analysis
4. **Process Correlation:** Track scores with processing parameters for optimization
5. **Application Focus:** Weight scales based on end-use requirements

## Example Results Interpretation

```
Scale 0: 0.95 → Excellent large-scale uniformity
Scale 1: 0.88 → Good fine texture uniformity  
Scale 2: 0.92 → Excellent medium-fine uniformity
Scale 3: 0.85 → Good medium-scale uniformity
Scale 4: 0.90 → Excellent coarse detail uniformity
Overall: 0.90 → Excellent overall uniformity
```
**Conclusion:** High-quality mat suitable for demanding applications

## Common Misconceptions

❌ **Wrong:** "Higher scale number = larger physical scale"  
✅ **Correct:** Scale 0 = largest, Scale 4 = finest details

❌ **Wrong:** "All scales should have equal weight"  
✅ **Correct:** Weight scales based on application requirements

❌ **Wrong:** "ROI should include entire image"  
✅ **Correct:** ROI should include only the mat region, excluding rulers/background

---
*For detailed explanations, see MULTISCALE_UNIFORMITY_GUIDE.md*
