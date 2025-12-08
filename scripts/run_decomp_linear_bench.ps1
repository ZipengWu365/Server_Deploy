param (
    [string]$CFG
)

$ErrorActionPreference = "Stop"

# Ensure package is installed/updated
Write-Host "Installing dependencies..."
pip install -e third_party/tsdecomp
pip install pyyaml pandas matplotlib seaborn scikit-learn

# Run experiment
Write-Host "Running experiment with config: $CFG"
python -m features.decomp_linear_bench.cli run --config $CFG

# Plot
# Parse out_dir from yaml (simple grep equivalent)
$OUT_DIR = Select-String -Path $CFG -Pattern "out_dir" | ForEach-Object { $_.Line.Split(":")[1].Trim() }
# Remove trailing slash if present for path joining, though python handles it usually.
# But let's be safe.
$OUT_DIR = $OUT_DIR.TrimEnd("/")
$OUT_DIR = $OUT_DIR.TrimEnd("\")

Write-Host "Plotting results to $OUT_DIR"
python -m features.decomp_linear_bench.cli plot --summary "$OUT_DIR/metrics_summary_by_method_horizon_ablation.csv" --out_dir $OUT_DIR
