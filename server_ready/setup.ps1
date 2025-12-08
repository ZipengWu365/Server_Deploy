# Quick Start Script for Windows PowerShell
# This script will guide you through the setup process

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Ridge + Decomposition Experiments Setup" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

# Check Python
Write-Host "1. Checking Python installation..." -ForegroundColor Yellow
try {
    $pythonVersion = python --version
    Write-Host "   ✓ $pythonVersion found" -ForegroundColor Green
} catch {
    Write-Host "   ✗ Python not found. Please install Python 3.8+" -ForegroundColor Red
    exit 1
}

# Install dependencies
Write-Host ""
Write-Host "2. Installing dependencies..." -ForegroundColor Yellow
pip install -r requirements.txt

# Check tsdecomp
Write-Host ""
Write-Host "3. Checking tsdecomp package..." -ForegroundColor Yellow
try {
    python -c "import tsdecomp" 2>$null
    Write-Host "   ✓ tsdecomp is installed" -ForegroundColor Green
} catch {
    Write-Host "   ✗ tsdecomp not found!" -ForegroundColor Red
    Write-Host "   Please install tsdecomp manually:" -ForegroundColor Yellow
    Write-Host "   - cd /path/to/tsdecomp"
    Write-Host "   - pip install -e ."
    Write-Host ""
    $choice = Read-Host "   Continue anyway? (y/n)"
    if ($choice -ne "y") {
        exit 1
    }
}

# Test setup
Write-Host ""
Write-Host "4. Testing environment..." -ForegroundColor Yellow
python scripts/test_experiment_setup.py

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "==========================================" -ForegroundColor Green
    Write-Host "✓ Setup complete! Ready to run experiments." -ForegroundColor Green
    Write-Host "==========================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "To start experiments:"
    Write-Host "  powershell -ExecutionPolicy Bypass -File scripts\run_all_ett_exchange_experiments.ps1"
    Write-Host ""
} else {
    Write-Host ""
    Write-Host "✗ Setup test failed. Please check the errors above." -ForegroundColor Red
}
