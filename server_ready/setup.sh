#!/bin/bash
# Quick Start Script for Linux/Mac
# This script will guide you through the setup process

echo "=========================================="
echo "Ridge + Decomposition Experiments Setup"
echo "=========================================="
echo ""

# Check Python
echo "1. Checking Python installation..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version)
    echo "   ✓ $PYTHON_VERSION found"
else
    echo "   ✗ Python 3 not found. Please install Python 3.8+"
    exit 1
fi

# Install dependencies
echo ""
echo "2. Installing dependencies..."
pip install -r requirements.txt

# Check tsdecomp
echo ""
echo "3. Checking tsdecomp package..."
if python3 -c "import tsdecomp" 2>/dev/null; then
    echo "   ✓ tsdecomp is installed"
else
    echo "   ✗ tsdecomp not found!"
    echo "   Please install tsdecomp manually:"
    echo "   - cd /path/to/tsdecomp"
    echo "   - pip install -e ."
    echo ""
    read -p "   Continue anyway? (y/n): " choice
    if [ "$choice" != "y" ]; then
        exit 1
    fi
fi

# Test setup
echo ""
echo "4. Testing environment..."
python3 scripts/test_experiment_setup.py

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✓ Setup complete! Ready to run experiments."
    echo "=========================================="
    echo ""
    echo "To start experiments:"
    echo "  bash scripts/run_all_ett_exchange_experiments.sh"
    echo ""
    echo "Or run in background:"
    echo "  nohup bash scripts/run_all_ett_exchange_experiments.sh > experiment_log.txt 2>&1 &"
    echo ""
else
    echo ""
    echo "✗ Setup test failed. Please check the errors above."
fi
