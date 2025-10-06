import importlib
import numpy as np
m = importlib.import_module('esnf_mat_analyzer.analysis.multiscale_uniformity')
print('Imported', m.MultiScaleUniformityAnalyzer)
an = m.MultiScaleUniformityAnalyzer()
th = np.zeros((16,16))
mask = np.zeros((16,16), dtype=bool)
mask[2:14,2:14] = True
res = an.analyze_multiscale_uniformity(th, mask)
print('keys', list(res.keys()))
