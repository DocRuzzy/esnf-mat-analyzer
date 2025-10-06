## Uniformity metrics for electrospun mat analysis

This guide documents the uniformity metrics computed by ESNF Mat Analyzer, their mathematical definitions, recommended interpretation thresholds, typical failure modes, and quick checks to include in tests. Use KaTeX-formatted formulas where helpful.

### Contract (inputs / outputs / error modes)
- Inputs: grayscale or background-corrected intensity image I (2D array), analysis mask M (boolean 2D array), optional spatial scale (pixels/mm).
- Outputs: dictionary of scalar metrics describing thickness and intensity uniformity (see list below).
- Errors: empty mask or zero-area mask; NaN/Inf in image; fully saturated image (all values near max). Functions should raise ValueError for empty masks and return NaN for metrics that cannot be computed.

### Common preprocessing steps
- If an image is color, convert to grayscale before computing intensity-based metrics.
- Apply background correction first when available; metrics computed on corrected images compare better across methods.
- Clip or ignore saturated pixels (>= 254 for uint8) when computing dynamic range-related metrics.

### Metrics

1) Mean thickness (or intensity) — \(\mu\)

\[\mu = \frac{1}{|M|} \sum_{(x,y)\in M} I(x,y)\]

Interpretation: central tendency of the signal inside the ROI. Edge cases: skewed by outliers and saturation.

2) Median thickness — \(\mathrm{median}(I[M])\)

Robust to outliers.

3) Standard deviation — \(\sigma\)

\[\sigma = \sqrt{\frac{1}{|M|-1} \sum_{(x,y)\in M} (I(x,y)-\mu)^2}\]

Interpretation: dispersion of intensity values. High \(\sigma\) indicates non-uniformity or remaining background.

4) Dynamic range (98th - 2nd percentile)

\[\mathrm{dyn} = P_{98}(I[M]) - P_{2}(I[M])\]

Less sensitive to extreme outliers; use percentiles to avoid single saturated pixels dominating.

5) Saturation percentage

\[\mathrm{saturation\_pct} = 100 \times \frac{\#\{I(x,y)\geq s\;|\; (x,y)\in M\}}{|M|}\]

Where \(s\) is a saturation threshold (default 254 for uint8). Values > 1-2% indicate potential measurement issues.

6) Coefficient of variation (CV)

\[\mathrm{CV} = \frac{\sigma}{\mu}\]

Normalized dispersion. Useful to compare across samples.

7) Local uniformity (sliding-window MAD or std)

Compute a local standard deviation or median absolute deviation in a small window (e.g., 15x15 pixels). Then compute the mean or percentile of the local std over the ROI. This metric detects local speckle or texture inhomogeneity.

8) Spatial autocorrelation length (optional)

Estimate by computing the 2D autocorrelation of intensities inside the masked ROI and finding the distance where correlation drops to 1/e. Gives a length-scale of texture variations (requires non-sparse ROIs).

9) Anisotropy / Orientation dispersion (FFT-based)

Compute the 2D FFT of the masked image (or gradients), collapse to polar coordinates, and assess the angular variance. Useful for aligned fiber mats.

### Recommended thresholds and interpretations
- CV &lt; 0.05: highly uniform
- CV 0.05–0.15: acceptable
- CV &gt; 0.15: non-uniform — investigate background correction or deposition issues
- saturation_pct &gt; 2%: likely measurement artifact or overexposure
- dyn (units of intensity): compare across samples after consistent background correction; low values may indicate underexposure or overly aggressive correction

These thresholds are heuristics — tune for your imaging setup.

### Quick tests to add
- Empty mask: functions should raise ValueError.
- All-saturated image: saturation_pct = 100, other metrics should return NaN or controlled values.
- Known synthetic gradient: validate that mean, dyn, and CV match expected values within tolerance.
- Local uniformity: apply a texture with known local variance and verify the local std measure.

### Implementation notes
- Use NumPy for vectorized computations; avoid full-image temporaries when possible for large images (process by tiles or downsample for some metrics).
- Use masked arrays (numpy.ma) to simplify ignoring masked pixels and saturated values.
- Provide both absolute metrics and normalized versions (per-mm) when spatial scale is available.

### Examples (pseudo-code)

```python
import numpy as np

def mean_intensity(img, mask):
    if mask.sum() == 0:
        raise ValueError("Empty mask")
    vals = img[mask]
    return float(np.nanmean(vals))

def dynamic_range(img, mask, lo_pct=2, hi_pct=98):
    vals = img[mask]
    return float(np.nanpercentile(vals, hi_pct) - np.nanpercentile(vals, lo_pct))
```

### References
- Include references to background-correction guide and research notes in `docs/`.

---

Document last updated: 2025-10-05
