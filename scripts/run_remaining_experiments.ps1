# 自动运行所有剩余实验
# Auto-run all remaining multi-scale experiments

Write-Host "="*80 -ForegroundColor Cyan
Write-Host "Multi-scale Decomposition - Batch Execution" -ForegroundColor Cyan
Write-Host "="*80 -ForegroundColor Cyan

$experiments = @(
    @{Name = "ETTh1 All Methods"; Config = "etth1_all.yaml"; Status = "Running" },
    @{Name = "ETTh2 All Methods"; Config = "etth2_all.yaml"; Status = "Pending" },
    @{Name = "ETTm1 All Methods"; Config = "ettm1_all.yaml"; Status = "Pending" },
    @{Name = "ETTm2 All Methods"; Config = "ettm2_all.yaml"; Status = "Pending" },
    @{Name = "Exchange Rate All"; Config = "exchange_all.yaml"; Status = "Pending" }
)

Write-Host "`nExperiment Queue:" -ForegroundColor Yellow
foreach ($exp in $experiments) {
    $status = if ($exp.Status -eq "Running") { "[RUNNING]" } else { "[QUEUED]" }
    $color = if ($exp.Status -eq "Running") { "Green" } else { "Gray" }
    Write-Host "  $status $($exp.Name)" -ForegroundColor $color
}

Write-Host "`n" -NoNewline
Write-Host "Waiting for ETTh1 to complete..." -ForegroundColor Yellow
Write-Host "Then will auto-run: ETTh2 -> ETTm1 -> ETTm2 -> Exchange" -ForegroundColor Gray

# Wait for etth1_all to complete (check every 5 minutes)
$etth1_output = "outputs/decomp_linear_bench/etth1_all_methods/metrics_summary_by_method_horizon_ablation.csv"

while (-not (Test-Path $etth1_output)) {
    Write-Host "." -NoNewline -ForegroundColor Gray
    Start-Sleep -Seconds 300  # 5 minutes
}

Write-Host "`n[COMPLETE] ETTh1 finished!" -ForegroundColor Green

# Run remaining experiments sequentially
$remaining = $experiments | Where-Object { $_.Status -eq "Pending" }
$current = 0
$total = $remaining.Count

foreach ($exp in $remaining) {
    $current++
    Write-Host "`n" -NoNewline
    Write-Host ("=" * 80) -ForegroundColor Cyan
    Write-Host ("[{0}/{1}] {2}" -f $current, $total, $exp.Name) -ForegroundColor Green
    Write-Host ("=" * 80) -ForegroundColor Cyan
    
    $configPath = "configs/decomp_linear/$($exp.Config)"
    
    try {
        python -m features.decomp_linear_bench.cli run --config $configPath
        Write-Host "[OK] $($exp.Name) completed" -ForegroundColor Green
    }
    catch {
        Write-Host "[ERROR] $($exp.Name) failed: $_" -ForegroundColor Red
    }
}

Write-Host "`n" -NoNewline
Write-Host ("=" * 80) -ForegroundColor Cyan
Write-Host "All experiments completed!" -ForegroundColor Cyan
Write-Host ("=" * 80) -ForegroundColor Cyan

# Summary
Write-Host "`nResults:" -ForegroundColor White
Get-ChildItem outputs/decomp_linear_bench -Recurse -Filter "metrics_summary*.csv" | 
ForEach-Object { 
    $size = [math]::Round($_.Length / 1KB, 2)
    Write-Host "  - $($_.Directory.Name): $size KB" -ForegroundColor Gray
}
