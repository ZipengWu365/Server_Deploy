# Run all ETT and Exchange experiments (PowerShell version)
# Usage: powershell -ExecutionPolicy Bypass -File scripts/run_all_ett_exchange_experiments.ps1

$ErrorActionPreference = "Stop"

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Starting ETT and Exchange Experiments" -ForegroundColor Cyan
Write-Host "Start time: $(Get-Date)" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan

# ETTh1
Write-Host ""
Write-Host "=== Running ETTh1 Baseline ===" -ForegroundColor Yellow
python -m features.decomp_linear_bench.cli run --config configs/decomp_linear/etth1_baseline.yaml

Write-Host ""
Write-Host "=== Running ETTh1 All Methods ===" -ForegroundColor Yellow
python -m features.decomp_linear_bench.cli run --config configs/decomp_linear/etth1_all.yaml

# ETTh2
Write-Host ""
Write-Host "=== Running ETTh2 Baseline ===" -ForegroundColor Yellow
python -m features.decomp_linear_bench.cli run --config configs/decomp_linear/etth2_baseline.yaml

Write-Host ""
Write-Host "=== Running ETTh2 All Methods ===" -ForegroundColor Yellow
python -m features.decomp_linear_bench.cli run --config configs/decomp_linear/etth2_all.yaml

# ETTm1
Write-Host ""
Write-Host "=== Running ETTm1 Baseline ===" -ForegroundColor Yellow
python -m features.decomp_linear_bench.cli run --config configs/decomp_linear/ettm1_baseline.yaml

Write-Host ""
Write-Host "=== Running ETTm1 All Methods ===" -ForegroundColor Yellow
python -m features.decomp_linear_bench.cli run --config configs/decomp_linear/ettm1_all.yaml

# ETTm2
Write-Host ""
Write-Host "=== Running ETTm2 Baseline ===" -ForegroundColor Yellow
python -m features.decomp_linear_bench.cli run --config configs/decomp_linear/ettm2_baseline.yaml

Write-Host ""
Write-Host "=== Running ETTm2 All Methods ===" -ForegroundColor Yellow
python -m features.decomp_linear_bench.cli run --config configs/decomp_linear/ettm2_all.yaml

# Exchange Rate
Write-Host ""
Write-Host "=== Running Exchange Rate Baseline ===" -ForegroundColor Yellow
python -m features.decomp_linear_bench.cli run --config configs/decomp_linear/exchange_baseline.yaml

Write-Host ""
Write-Host "=== Running Exchange Rate All Methods ===" -ForegroundColor Yellow
python -m features.decomp_linear_bench.cli run --config configs/decomp_linear/exchange_all.yaml

Write-Host ""
Write-Host "==========================================" -ForegroundColor Green
Write-Host "All experiments completed!" -ForegroundColor Green
Write-Host "End time: $(Get-Date)" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Green
Write-Host ""
Write-Host "Results saved in:"
Write-Host "  - outputs/decomp_linear_bench/baseline/"
Write-Host "  - outputs/decomp_linear_bench/all_methods/"
Write-Host ""
Write-Host "To analyze results, run:"
Write-Host "  python scripts/analyze_results.py"
