# FF-VUS

**FF-VUS (Furiously Fast Volume Under Surface)** is a lightning-fast implementation of the next-generation metric for **time series anomaly detection**: **VUS-PR** — _Volume Under the Precision-Recall Surface_. This metric provides a more comprehensive view of detection performance, especially suited for tasks with **imprecise anomaly boundaries** and **gradual transitions**.

---

## 🚀 Overview

Traditional metrics like AUC-PR or F1-score collapse the evaluation into single values that may ignore the **spatial tolerance** needed in time series anomaly detection. VUS-PR addresses this by evaluating across different slope sizes around ground-truth anomalies, producing a **3D surface** over:

- **Slope tolerance** (how much leeway we give before and after an anomaly),
- **Precision** and **Recall** values,
- And integrating this surface to get a scalar value.

FF-VUS implements this efficiently, making it scalable for **large datasets** and **massive experiments**.

---

## 📦 Features

- ✅ Blazing fast VUS-PR computation on large-scale datasets (NumPy and GPU-accelerated Torch backends)
- 🧠 Smart slope expansion around anomaly intervals
- 🔁 Flexible configuration of slope size, step, and existence/confusion matrix modes
- 🛠️ Minimal dependencies (NumPy, SciPy, Torch for GPU)
- 📊 Generates realistic anomaly scores with noise, detection uncertainty, and false positives
- 🧪 Synthetic data generation for benchmarking
- 📈 Jupyter notebooks for analysis and visualization

---

## 📂 Project Structure

```
ff-vus/
├── src/
│   ├── compute_metric.py         # Main metric computation entry point
│   ├── generate_synthetic.py     # Synthetic data and score generation
│   ├── vus/
│   │   ├── vus_numpy.py          # NumPy implementation of VUS-PR
│   │   └── vus_torch.py          # Torch (GPU) implementation of VUS-PR
│   ├── utils/
│   │   ├── utils.py              # Helpers for timing, labeling, plotting, etc.
│   │   └── metricloader.py       # Metric I/O utilities
│   └── legacy/                   # Legacy metrics and models for comparison
├── notebooks/                    # Example and analysis notebooks
├── data/                         # Input/output data
├── experiments/                  # Generated results and logs
└── README.md
```

---

## 🧪 Generating Synthetic Data

You can simulate labeled time series data and corresponding anomaly scores using:

```python
from src.generate_synthetic import generate_synthetic_labels, generate_score_from_labels
```

Synthetic score generation includes:

- **Lag and noise** injection
- **Probabilistic detection**
- **Gamma-shaped anomaly bumps**
- **False positives** at controlled rates

To generate a batch of synthetic datasets:

```python
from src.generate_synthetic import generate_synthetic

generate_synthetic(
    n_timeseries=10,
    ts_length=1000,
    n_anomalies=10,
    avg_anomaly_length=100,
    file_type='npy'
)
```

---

## ⚙️ VUS Parameters

You can control how VUS is computed with several parameters:

- `slope_size`: Maximum number of time steps added before/after anomalies as tolerance.
- `step`: Step size used when incrementing slope (must divide `slope_size`).
- `slopes`: Slope computation mode (`'precomputed'` or `'function'`)
- `existence`: Existence computation mode (`'optimized'`, `'matrix'`, `'trivial'`, or `'None'`)
- `conf_matrix`: Confusion matrix computation mode (`'dynamic'`, etc.)

For example:

```python
from src.vus.vus_numpy import VUSNumpy

vus = VUSNumpy(slope_size=50, step=5, slopes='precomputed', existence='optimized')
vus_value, timing = vus.compute(label, score)
```

Or for GPU acceleration:

```python
from src.vus.vus_torch import VUSTorch

vus = VUSTorch(slope_size=50, step=5, device='cuda')
vus_value, timing = vus.compute(label, score)
```

---

## 📈 Example Usage

Generate 100 labeled time series and compute anomaly scores:

```python
from src.generate_synthetic import generate_synthetic_labels, generate_score_from_labels

n_labels = 100
length = 10_000
labels = []
scores = []
for _ in range(n_labels):
    label, start_points, end_points = generate_synthetic_labels(length, n_anomalies=10, avg_anomaly_length=100)
    score = generate_score_from_labels(label, start_points, end_points)
    labels.append(label)
    scores.append(score)
```

Compute VUS:

```python
from src.vus.vus_numpy import VUSNumpy

vus = VUSNumpy(slope_size=50, step=5)
vus_values = [vus.compute(label, score)[0] for label, score in zip(labels, scores)]
```

---

## 📌 Requirements

- Python 3.8+
- NumPy
- SciPy
- tqdm (optional, for progress bars)
- torch (optional, for GPU acceleration)
- seaborn, matplotlib (for plotting/visualization)

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 📤 Output

All generated labels, scores, and VUS values are saved under the `experiments/` directory with filenames encoding the parameter setup.

---

## 💡 Naming

Yes, FF-VUS stands for _Furiously Fast VUS_. But also... **Freaking Fast**, because we can 😎

---

## 🧑‍💻 Authors

- **Emmanouil Sylligardos** – PhD Researcher @ École Normale Supérieure
- **Paul Boniol** – Researcher @ École Normale Supérieure

---

## 📜 License

MIT License – do whatever you want, just don't forget to cite us if it helps you!
