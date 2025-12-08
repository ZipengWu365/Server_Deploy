# 批量运行所有多尺度实验
# Run all multi-scale decomposition experiments

Write-Host "=" -ForegroundColor Cyan -NoNewline
Write-Host ("="*79) -ForegroundColor Cyan
Write-Host "Multi-scale Decomposition Benchmark - Batch Execution" -ForegroundColor Cyan
Write-Host ("="*80) -ForegroundColor Cyan

$experiments = @(
    @{Name="Quick Multiscale (Already Done)"; Config="quick_multiscale.yaml"; Skip=$true},
    @{Name="Compare Multiscale Methods"; Config="compare_multiscale_methods.yaml"; Skip=$false},
    @{Name="ETTh1 All Methods"; Config="etth1_all.yaml"; Skip=$false},
    @{Name="ETTh2 All Methods"; Config="etth2_all.yaml"; Skip=$false},
    @{Name="ETTm1 All Methods"; Config="ettm1_all.yaml"; Skip=$false},
    @{Name="ETTm2 All Methods"; Config="ettm2_all.yaml"; Skip=$false},
    @{Name="Exchange Rate All"; Config="exchange_all.yaml"; Skip=$false}
)

$total = ($experiments | Where-Object {-not $_.Skip}).Count
$current = 0

foreach ($exp in $experiments) {
    if ($exp.Skip) {
        Write-Host "`n[SKIP] $($exp.Name)" -ForegroundColor Yellow
        continue
    }
    
    $current++
    Write-Host "`n" -NoNewline
    Write-Host ("[{0}/{1}] " -f $current, $total) -ForegroundColor Green -NoNewline
    Write-Host $exp.Name -ForegroundColor White
    Write-Host ("─"*80) -ForegroundColor DarkGray
    
    $configPath = "configs/decomp_linear/$($exp.Config)"
    
    try {
        python -m features.decomp_linear_bench.cli run --config $configPath
        Write-Host "[OK] $($exp.Name) completed" -ForegroundColor Green
    } catch {
        Write-Host "[ERROR] $($exp.Name) failed: $_" -ForegroundColor Red
    }
}

Write-Host "`n" -NoNewline
Write-Host ("="*80) -ForegroundColor Cyan
Write-Host "All experiments completed!" -ForegroundColor Cyan
Write-Host ("="*80) -ForegroundColor Cyan

# Summary
Write-Host "`nResults saved in:" -ForegroundColor White
Get-ChildItem outputs/decomp_linear_bench -Recurse -Filter "metrics_summary*.csv" | 
    ForEach-Object { Write-Host "  - $($_.FullName)" -ForegroundColor Gray }

