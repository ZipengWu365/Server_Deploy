# Run all baseline experiments
$configs = @(
    "etth1_baseline.yaml",
    "etth2_baseline.yaml",
    "ettm1_baseline.yaml",
    "ettm2_baseline.yaml",
    "exchange_baseline.yaml"
)

$startTime = Get-Date
Write-Host "=" * 80
Write-Host "Running All Baseline Experiments"
Write-Host "Start Time: $startTime"
Write-Host "=" * 80

foreach ($cfg in $configs) {
    Write-Host "`n>>> Running: $cfg"
    python -m features.decomp_linear_bench.cli run --config "configs/decomp_linear/$cfg"
}

$endTime = Get-Date
$duration = $endTime - $startTime
Write-Host "`n" + "=" * 80
Write-Host "All experiments completed!"
Write-Host "End Time: $endTime"
Write-Host "Total Duration: $duration"
Write-Host "=" * 80
