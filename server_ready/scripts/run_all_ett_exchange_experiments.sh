#!/bin/bash
# Run all ETT and Exchange experiments
# Usage: bash scripts/run_all_ett_exchange_experiments.sh

set -e  # Exit on error

echo "=========================================="
echo "Starting ETT and Exchange Experiments"
echo "Start time: $(date)"
echo "=========================================="

# ETTh1
echo ""
echo "=== Running ETTh1 Baseline ==="
python -m features.decomp_linear_bench.cli run --config configs/decomp_linear/etth1_baseline.yaml

echo ""
echo "=== Running ETTh1 All Methods ==="
python -m features.decomp_linear_bench.cli run --config configs/decomp_linear/etth1_all.yaml

# ETTh2
echo ""
echo "=== Running ETTh2 Baseline ==="
python -m features.decomp_linear_bench.cli run --config configs/decomp_linear/etth2_baseline.yaml

echo ""
echo "=== Running ETTh2 All Methods ==="
python -m features.decomp_linear_bench.cli run --config configs/decomp_linear/etth2_all.yaml

# ETTm1
echo ""
echo "=== Running ETTm1 Baseline ==="
python -m features.decomp_linear_bench.cli run --config configs/decomp_linear/ettm1_baseline.yaml

echo ""
echo "=== Running ETTm1 All Methods ==="
python -m features.decomp_linear_bench.cli run --config configs/decomp_linear/ettm1_all.yaml

# ETTm2
echo ""
echo "=== Running ETTm2 Baseline ==="
python -m features.decomp_linear_bench.cli run --config configs/decomp_linear/ettm2_baseline.yaml

echo ""
echo "=== Running ETTm2 All Methods ==="
python -m features.decomp_linear_bench.cli run --config configs/decomp_linear/ettm2_all.yaml

# Exchange Rate
echo ""
echo "=== Running Exchange Rate Baseline ==="
python -m features.decomp_linear_bench.cli run --config configs/decomp_linear/exchange_baseline.yaml

echo ""
echo "=== Running Exchange Rate All Methods ==="
python -m features.decomp_linear_bench.cli run --config configs/decomp_linear/exchange_all.yaml

echo ""
echo "=========================================="
echo "All experiments completed!"
echo "End time: $(date)"
echo "=========================================="
echo ""
echo "Results saved in:"
echo "  - outputs/decomp_linear_bench/baseline/"
echo "  - outputs/decomp_linear_bench/all_methods/"
echo ""
echo "To analyze results, run:"
echo "  python scripts/analyze_results.py"
