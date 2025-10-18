#!/bin/bash
#
# setup_server.sh
#
# This script automates the setup process for the SSA-CNN forecasting model
# on a new Linux server with CUDA.
#
# It performs the following steps:
# 1. Defines the project structure.
# 2. Checks for required directories (FastTime, third_party).
# 3. Creates and activates a Python virtual environment.
# 4. Installs necessary Python packages from requirements.
# 5. Compiles and installs the 'fasttimes' C++ extension.
# 6. Provides instructions on how to run the main script.

set -e # Exit immediately if a command exits with a non-zero status.

# --- Configuration ---
VENV_NAME="ssacnn_env"
PYTHON_EXE="python3.10" # Feel free to change this to python3.9, python3.11 etc.

# --- Project Structure Validation ---
echo "--- [Step 1/5] Verifying project structure ---"
if [ ! -d "FastTime/fasttimes-package" ] || [ ! -d "third_party" ]; then
    echo "❌ Error: Missing required directories."
    echo "Please ensure 'FastTime/fasttimes-package/' and 'third_party/' are in the same directory as this script."
    exit 1
fi
echo "✅ Project structure is valid."
echo ""

# --- Virtual Environment Setup ---
echo "--- [Step 2/5] Creating Python virtual environment ---"
if [ -d "$VENV_NAME" ]; then
    echo "ℹ️ Virtual environment '$VENV_NAME' already exists. Skipping creation."
else
    $PYTHON_EXE -m venv $VENV_NAME
    echo "✅ Virtual environment created."
fi
source $VENV_NAME/bin/activate
echo "✅ Virtual environment activated."
echo ""

# --- Python Package Installation ---
echo "--- [Step 3/5] Installing Python packages ---"
# Install core dependencies. PyTorch is installed separately for CUDA.
pip install numpy pandas scikit-learn psutil
# Install PyTorch with CUDA 11.8 support.
# For other CUDA versions, find the correct command at https://pytorch.org/get-started/previous-versions/
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
echo "✅ Python packages installed."
echo ""

# --- C++ Extension Compilation & Installation ---
echo "--- [Step 4/5] Compiling and installing 'fasttimes' C++ extension ---"
# The C++ extension requires pybind11, which will be installed by the setup script.
# The EIGEN3_INCLUDE_DIR environment variable tells the compiler where to find the Eigen library headers.
export EIGEN3_INCLUDE_DIR="../third_party/eigen-3.4.0"
cd FastTime/fasttimes-package

# Clean previous builds if any
python setup.py clean --all

# Compile and install
echo "Building and installing... This may take a few minutes."
pip install .
cd ../.. # Return to the root project directory
echo "✅ 'fasttimes' extension successfully installed."
echo ""


# --- Final Instructions ---
echo "--- [Step 5/5] Setup Complete! ---"
echo "✅ Environment is ready."
echo ""
echo "To run the training, use the following command:"
echo "source $VENV_NAME/bin/activate"
echo "python ssa_cnn_server_optimized.py"
echo ""
