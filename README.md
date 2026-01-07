# FF-VUS

FF-VUS provides efficient implementations of the Volume Under the Precision-Recall Surface (VUS-PR) for time series anomaly detection. This repository accompanies the paper "FF-VUS: A Freaking-Fast Evaluation Measure for Time Series Anomaly Detection" and contains the exact CPU algorithm (FF-VUS) and a fully vectorized GPU approximation (FF-VUS-GPU).

## Summary

VUS-PR is a threshold-independent metric that evaluates detection quality across slope tolerances, producing a 3D precision–recall surface whose volume summarizes performance. FF-VUS removes computational bottlenecks through algorithmic optimizations and a GPU-parallel approximation, enabling practical evaluation at large scale.

## Key contributions

- FF-VUS: an exact, CPU-based algorithm that avoids redundant computations.
- FF-VUS-GPU: a vectorized approximation exploiting GPU parallelism for massive speedups.
- Extensive experiments on the TSB-UAD benchmark and synthetic series up to billions of points demonstrating orders-of-magnitude speed improvements.

Representative results (paper)

- Average speed-up of FF-VUS over baseline VUS: ~111×
- Average speed-up of FF-VUS-GPU over baseline VUS: ~467×
- Extreme scenario: benchmark evaluation reduced from ~61 hours to ~2 minutes (>1800× improvement)

## Repository contents

- src/: metric implementations, synthetic data generators, utilities
  - src/vus/vus_numpy.py — FF-VUS (CPU)
  - src/vus/vus_torch.py — FF-VUS-GPU (vectorized, optional)
  - src/generate_synthetic.py — synthetic data & score generation
  - src/utils/ — helpers for timing, I/O and plotting
  - src/notebooks/: analysis and visualization notebooks used in the paper
- experiments/: generated results and figures

## Requirements & installation

Python 3.12+, with (minimum):

- numpy, scipy, tqdm
- matplotlib, seaborn (optional, for plotting)
- torch (optional, for GPU acceleration)

Install:

```bash
pip install -r requirements.txt
```

## Minimal usage

Compute VUS (CPU):

```python
from src.vus.vus_numpy import VUSNumpy
vus = VUSNumpy(
  slope_size=128                    # The buffer length that we add on the left and right of the label
  step=1,                           # The step for which we repeat the buffers, e.g. slope_size = 100 and step = 1 results to 100 buffers but if step = 5 it results to 20 buffers
  zita=(1/math.sqrt(2)),            # The height on the y-axis at which the buffer reaches, the higher this value, the higher the reward
  global_mask=True,                 # Optimization option which is very helpful in very long time series
  slopes='precomputed',             # The way we compute the buffers, please leave this to the default option for every version, rest of the modes were used for cross evaluation while designing
  existence='optimized',            # The way we compute the existence reward, same as for 'slopes'
  conf_matrix='dynamic_plus',       # The optimization level for the matrix, same as for 'slopes'
  interpolation='stepwise',         # We have 2 different modes for interpolating the final surface, although the default is the one we should mainly use
  metric='vus_pr',                  # This option is not implemented yet and it won't affect anything, in the feature we may compute more measures
  side='both',                      # The side to which we would like to add the buffers, e.g. for streaming applications, it only makes sense to add the buffers after the event not before
)
value, timing = vus.compute(label, score)
```

Compute VUS (GPU, optional):

```python
from src.vus.vus_torch import VUSTorch
vus = VUSTorch(slope_size=128, device='cuda')
value, timing = vus.compute(label, score)
```

## Experiments

Notebooks under src/notebooks reproduce figures and runtime analyses reported in the paper. Results and plotting scripts save outputs to experiments/figures/.

## Citation

If you use this work, please cite the paper: "FF-VUS: A Freaking-Fast Evaluation Measure for Time Series Anomaly Detection." (under review)

## Authors & License

Authors: Emmanouil Sylligardos, Paul Boniol, John Paparrizos, Pierre Senellart  
License: MIT
