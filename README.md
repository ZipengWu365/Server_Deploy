# Time Series Decomposition + Ridge Regression Benchmark

**A comprehensive benchmarking framework for evaluating time series decomposition methods combined with Ridge regression for long-term forecasting.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

---

## 🎯 Overview

This repository provides a complete experimental framework for benchmarking time series decomposition methods (STL, SSA, EMD, Wavelet, STD_MULTI, Moving Average) combined with Ridge regression on standard long-term forecasting datasets.

### Key Features

- ✅ **6 Decomposition Methods**: STL, SSA, EMD, Wavelet, STD_MULTI, MA_BASELINE
- ✅ **5 Standard Datasets**: ETTh1, ETTh2, ETTm1, ETTm2, Exchange Rate
- ✅ **Multiple Horizons**: 96, 192, 336, 720 timesteps
- ✅ **Ablation Studies**: Trend (+T), Seasonal (+S), Trend+Seasonal (+TS)
- ✅ **Data Leakage Fixed**: Per-window decomposition ensures no future information leakage
- ✅ **One-Click Execution**: Automated scripts for batch experiments
- ✅ **Comprehensive Analysis**: Automatic result aggregation and comparison

---

## 📦 Project Structure

```
Server_Deploy/
├── features/
│   └── decomp_linear_bench/     # Core experiment engine
│       ├── cli.py               # Command-line interface
│       ├── runner.py            # Experiment runner (data leakage fixed)
│       ├── builder.py           # Feature construction
│       ├── ablations.py         # Component selection logic
│       ├── learners.py          # Ridge regression models
│       └── report.py            # Result reporting
│
├── configs/
│   └── decomp_linear/           # Experiment configurations (YAML)
│       ├── EXPERIMENT_DESIGN.md # Experimental design document
│       ├── etth1_baseline.yaml  # Baseline experiments
│       ├── etth1_all.yaml       # All decomposition methods
│       └── ...                  # (10 configs total)
│
├── scripts/
│   ├── run_all_ett_exchange_experiments.sh   # Batch runner (Linux/Mac)
│   ├── run_all_ett_exchange_experiments.ps1  # Batch runner (Windows)
│   ├── test_experiment_setup.py              # Quick validation test
│   └── analyze_results.py                    # Result analysis tool
│
├── dataset/                     # Time series datasets (CSV)
│   ├── ETTh1.csv               # Electricity Transformer Temperature (hourly)
│   ├── ETTh2.csv
│   ├── ETTm1.csv               # (15-minute frequency)
│   ├── ETTm2.csv
│   └── exchange_rate.csv       # Daily exchange rates
│
├── third_party/
│   ├── tsdecomp/               # Custom decomposition package
│   └── eigen-3.4.0/            # C++ linear algebra library
│
├── server_ready/               # Standalone deployment package
│   ├── README.md
│   ├── requirements.txt
│   ├── setup.sh / setup.ps1
│   └── ...                     # (subset of main repo)
│
└── archive/                    # Legacy experiments and results
```

---

## 🚀 Quick Start

### 1. Environment Setup

**Prerequisites:**
- Python 3.8 or higher
- pip package manager
- Git

**Install Dependencies:**

```bash
# Clone the repository
git clone https://github.com/ZipengWu365/Server_Deploy.git
cd Server_Deploy

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install Python packages
pip install -r server_ready/requirements.txt

# Install tsdecomp package (critical dependency)
pip install -e third_party/tsdecomp
```

### 2. Quick Test

Run a quick test to verify the setup:

```bash
python scripts/test_experiment_setup.py
```

Expected output:
```
✅ Baseline test passed!
✅ STL decomposition test passed!
```

### 3. Run Experiments

**Option A: Run All Experiments (Recommended)**

Linux/Mac:
```bash
bash scripts/run_all_ett_exchange_experiments.sh
```

Windows PowerShell:
```powershell
.\scripts\run_all_ett_exchange_experiments.ps1
```

**Option B: Run Individual Experiments**

```bash
# Baseline experiment (raw features only)
python -m features.decomp_linear_bench.cli run --config configs/decomp_linear/etth1_baseline.yaml

# All decomposition methods
python -m features.decomp_linear_bench.cli run --config configs/decomp_linear/etth1_all.yaml
```

**Estimated Runtime:**
- Single dataset (baseline + all methods): ~40-60 minutes
- All 5 datasets: ~4 hours (on standard CPU)

### 4. Analyze Results

After experiments complete:

```bash
python scripts/analyze_results.py
```

This generates multiple reports in `outputs/decomp_linear_bench/analysis/`:
- `summary_by_dataset.csv` - Performance by dataset
- `summary_by_method.csv` - Performance by decomposition method
- `best_configs.csv` - Best performing configurations
- `improvement_over_baseline.csv` - Improvement percentages
- And more...

---

## 📊 Datasets

| Dataset | Frequency | Channels | Length | Domain | Included |
|---------|-----------|----------|--------|--------|----------|
| ETTh1 | Hourly | 7 | 17,420 | Electricity Transformer Temperature | ✅ |
| ETTh2 | Hourly | 7 | 17,420 | Electricity Transformer Temperature | ✅ |
| ETTm1 | 15-min | 7 | 69,680 | Electricity Transformer Temperature | ✅ |
| ETTm2 | 15-min | 7 | 69,680 | Electricity Transformer Temperature | ✅ |
| Exchange | Daily | 8 | 7,588 | Currency Exchange Rates | ✅ |
| Electricity | Hourly | 321 | 26,304 | Electricity consumption | ⚠️ Too large (91 MB) |
| Traffic | Hourly | 862 | 17,544 | Road occupancy rates | ⚠️ Too large (130 MB) |

**Note:** Electricity and Traffic datasets are excluded from the repository due to GitHub's 100MB file size limit. They are available in the main `dataset/` folder for local use, but not in `server_ready/`.

**Train/Val/Test Split:** Standard 70% / 10% / 20% split (configurable via YAML)

---

## 🧪 Experimental Design

### Decomposition Methods

1. **STL** (Seasonal-Trend decomposition using LOESS)
2. **SSA** (Singular Spectrum Analysis)
3. **EMD** (Empirical Mode Decomposition)
4. **Wavelet** (Discrete Wavelet Transform)
5. **STD_MULTI** (Multi-scale Seasonal-Trend Decomposition)
6. **MA_BASELINE** (Simple Moving Average)

### Ablation Studies

For each decomposition method, we test:
- **+T**: Trend component only
- **+S**: Seasonal component only
- **+TS**: Both trend and seasonal components

### Evaluation Metrics

- **MAE** (Mean Absolute Error)
- **MSE** (Mean Squared Error)
- **MAPE** (Mean Absolute Percentage Error)

### Forecasting Horizons

- 96 timesteps
- 192 timesteps
- 336 timesteps
- 720 timesteps

Lookback window: **336 timesteps** (fixed)

---

## 🔧 Configuration

Each experiment is defined by a YAML configuration file in `configs/decomp_linear/`:

```yaml
experiment_name: "etth1_baseline"
output_dir: "outputs/decomp_linear_bench/etth1"

dataset:
  name: "ETTh1"
  path: "dataset/ETTh1.csv"
  split: "standard"  # 70/10/20 split

decomp_methods:
  - "NONE"  # Baseline (no decomposition)

horizons: [96, 192, 336, 720]
lookback: 336

ablation_components:
  - "RAW"  # Raw features without decomposition

learner:
  type: "Ridge"
  alpha: 1.0
```

**Customization:**
- Edit YAML files to change datasets, methods, or hyperparameters
- Create new configs for custom experiments
- See `EXPERIMENT_DESIGN.md` for detailed design rationale

---

## 🐛 Important: Data Leakage Fix

**Critical Issue Fixed:** Earlier versions decomposed entire train/test sequences once, which allowed decomposition algorithms to use future information when estimating components for current timesteps.

**Solution:** `runner.py` now decomposes each sliding window independently, ensuring only past data (lookback window) is used for decomposition at each prediction step.

**Impact:**
- ✅ Eliminates future information leakage
- ✅ Provides fair comparison across methods
- ⏱️ Increases runtime (~2x slower, necessary trade-off)

---

## 📈 Results Location

All experimental results are saved to:

```
outputs/
└── decomp_linear_bench/
    ├── etth1/
    │   ├── baseline/
    │   │   ├── metrics_summary_by_method_horizon_ablation.csv
    │   │   └── detailed_metrics.csv
    │   └── all_methods/
    │       └── ...
    ├── etth2/
    ├── ettm1/
    ├── ettm2/
    ├── exchange/
    └── analysis/          # Aggregated analysis (from analyze_results.py)
        ├── summary_by_dataset.csv
        ├── summary_by_method.csv
        └── ...
```

---

## 🖥️ Server Deployment

For clean server deployment, use the `server_ready/` folder which contains only essential files:

```bash
cd server_ready/

# Setup (Linux/Mac)
bash setup.sh

# Setup (Windows)
.\setup.ps1

# Run experiments
bash scripts/run_all_ett_exchange_experiments.sh
```

See `server_ready/README.md` for detailed deployment instructions.

---

## 🔍 Troubleshooting

### Issue: `ModuleNotFoundError: No module named 'tsdecomp'`

**Solution:** Install the custom tsdecomp package:
```bash
pip install -e third_party/tsdecomp
# Or add to PYTHONPATH:
export PYTHONPATH="${PYTHONPATH}:$(pwd)/third_party/tsdecomp"
```

### Issue: `FileNotFoundError: dataset/ETTh1.csv not found`

**Solution:** Ensure you're running commands from the repository root:
```bash
cd /path/to/Server_Deploy
python -m features.decomp_linear_bench.cli run --config configs/decomp_linear/etth1_baseline.yaml
```

### Issue: Out of Memory

**Solution:** Reduce batch processing or run experiments sequentially:
- Edit YAML configs to test fewer horizons
- Run one dataset at a time instead of batch script

### Issue: Slow Execution

**Solution:** 
- Use faster decomposition methods (STL, MA_BASELINE are fastest)
- Reduce number of horizons in YAML config
- Use multi-core processing (currently single-threaded)

---

## 📚 Documentation

- **`EXPERIMENT_DESIGN.md`** - Detailed experimental design and rationale
- **`server_ready/RUN_EXPERIMENTS.md`** - Step-by-step execution guide
- **`server_ready/CHECKLIST.md`** - Pre-deployment checklist
- **`server_ready/QUICK_REF.md`** - Quick command reference

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

**Areas for Contribution:**
- Additional decomposition methods
- More datasets
- Parallel processing optimization
- Visualization tools
- Deep learning baselines

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 📧 Contact

**Repository Owner:** ZipengWu365  
**GitHub:** [https://github.com/ZipengWu365/Server_Deploy](https://github.com/ZipengWu365/Server_Deploy)

---

## 🙏 Acknowledgments

- ETT datasets from the [Time-Series-Library](https://github.com/thuml/Time-Series-Library)
- Decomposition methods from statsmodels, PyEMD, PyWavelets
- Ridge regression from scikit-learn

---

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@misc{server_deploy2024,
  title={Time Series Decomposition + Ridge Regression Benchmark},
  author={Wu, Zipeng},
  year={2024},
  publisher={GitHub},
  url={https://github.com/ZipengWu365/Server_Deploy}
}
```

---

**Last Updated:** December 7, 2025  
**Version:** 1.0.0
