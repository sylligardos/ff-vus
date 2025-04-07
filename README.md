# FF-VUS

**FF-VUS (Furiously Fast Volume Under Surface)** is a lightning-fast implementation of the next-generation metric for **time series anomaly detection**: **VUS-PR** — *Volume Under the Precision-Recall Surface*. This metric provides a more comprehensive view of detection performance, especially suited for tasks with **imprecise anomaly boundaries** and **gradual transitions**.

---

## 🚀 Overview

Traditional metrics like AUC-PR or F1-score collapse the evaluation into single values that may ignore the **spatial tolerance** needed in time series anomaly detection. VUS-PR addresses this by evaluating across different slope sizes around ground-truth anomalies, producing a **3D surface** over:

- **Slope tolerance** (how much leeway we give before and after an anomaly),
- **Precision** and **Recall** values,
- And integrating this surface to get a scalar value.

FF-VUS implements this efficiently, making it scalable for **large datasets** and **massive experiments**.

---

## 📦 Features

- ✅ Blazing fast VUS-PR computation on large-scale datasets
- 🧠 Smart slope expansion around anomaly intervals
- 🔁 Flexible configuration of slope size and step
- 🛠️ Minimal dependencies (NumPy, SciPy)
- 📊 Generates realistic anomaly scores with noise, detection uncertainty, and false positives

---

## 📂 Project Structure

```
ff-vus/
├── src/                 # Core implementation
│   ├── vus.py           # VUS computation logic
│   └── utils.py         # Helpers for timing, labeling, etc.
├── scripts/             # Scripts for running experiments
├── data/                # Input/output data
├── experiments/         # Generated results
└── README.md            # You are here!
```

---

## 🧪 Generating Synthetic Data

You can simulate labeled time series data and corresponding anomaly scores using:

```python
from src.synthetic import generate_synthetic_labels, generate_score_from_labels
```

Synthetic score generation includes:
- **Lag and noise** injection
- **Probabilistic detection**
- **Gamma-shaped anomaly bumps**
- **False positives** at controlled rates

---

## ⚙️ VUS Parameters

You can control how VUS is computed with two key parameters:

- `slope_size`: Maximum number of time steps added before/after anomalies as tolerance.
- `step`: Step size used when incrementing slope (must divide `slope_size`).

For example:
```python
vus = compute_vus(score, labels, slope_size=50, step=5)
```

---

## 📈 Example Usage

Generate 100 labeled time series and compute anomaly scores:
```python
n_labels = 100
length = 10_000_000
labels, scores = generate_dataset(n_labels, length)
```

Compute VUS:
```python
vus_values = [compute_vus(score, label, slope_size=50, step=5) for score, label in zip(scores, labels)]
```

---

## 📌 Requirements

- Python 3.8+
- NumPy
- SciPy
- tqdm (optional, for progress bars)

Install dependencies:
```bash
pip install -r requirements.txt
```

---

## 📤 Output

All generated labels, scores, and VUS values are saved under the `experiments/` directory with filenames encoding the parameter setup.

---

## 💡 Naming

Yes, FF-VUS stands for *Furiously Fast VUS*. But also... **Freaking Fast**, because we can 😎

---

## 🧑‍💻 Authors

- **Emmanouil Sylligardos** – PhD Researcher @ École Normale Supérieure
- **Paul Boniol** – Researcher @ École Normale Supérieure

---

## 📜 License

MIT License – do whatever you want, just don't forget to cite us if it helps you!
