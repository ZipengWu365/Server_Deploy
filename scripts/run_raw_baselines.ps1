# 批量运行 RAW Baseline 实验
# 对比纯 Ridge 回归 vs 分解方法

Write-Host "="*80 -ForegroundColor Cyan
Write-Host "RAW Baseline Experiments - Pure Ridge Regression" -ForegroundColor Cyan
Write-Host "="*80 -ForegroundColor Cyan

$datasets = @(
    "etth1_raw_baseline.yaml",
    "etth2_raw_baseline.yaml"
)

$total = $datasets.Count
$current = 0

foreach ($config in $datasets) {
    $current++
    Write-Host "`n[{0}/{1}] Running {2}..." -f $current, $total, $config -ForegroundColor Green
    
    try {
        python -m features.decomp_linear_bench.cli run --config "configs/decomp_linear/$config"
        Write-Host "[OK] $config completed" -ForegroundColor Green
    }
    catch {
        Write-Host "[ERROR] $config failed: $_" -ForegroundColor Red
    }
}

Write-Host "`n" -NoNewline
Write-Host ("=" * 80) -ForegroundColor Cyan
Write-Host "Baseline experiments completed!" -ForegroundColor Cyan
Write-Host ("=" * 80) -ForegroundColor Cyan

# Compare with decomposition methods
Write-Host "`nComparison Summary:" -ForegroundColor Yellow
Write-Host "  - RAW baselines saved in outputs/decomp_linear_bench/*_raw_baseline/" -ForegroundColor Gray
Write-Host "  - Compare with decomposition results to measure improvement" -ForegroundColor Gray
