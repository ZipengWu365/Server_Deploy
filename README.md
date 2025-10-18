# Server Deploy

This directory contains the deployment assets for the SSA-CNN server, including the setup script, models, and required datasets.

## Deployment Steps

1. Copy this repository to the target server.
2. Make the setup script executable:
   ```bash
   chmod +x setup_server.sh
   ```
3. Run the setup script to configure the environment:
   ```bash
   ./setup_server.sh
   ```

The script installs dependencies, arranges dataset files, and launches the optimized SSA-CNN service.

## Repository Structure

- `setup_server.sh`: Automated server configuration and launch script.
- `FastTime/` and `third_party/`: Supporting libraries and accelerated implementations.
- `dataset/`: Time-series datasets required by the service.
- `ssa_cnn_server_optimized.py`: Entry point for serving the SSA-CNN model.

Make changes as needed for your infrastructure, then rerun the setup script to apply them.
