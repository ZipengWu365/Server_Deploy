# Server Deploy

部署之后要快速使用，只需在服务器上依次执行：

```bash
chmod +x setup_server.sh
./setup_server.sh
```

The script installs dependencies, arranges dataset files, and launches the optimized SSA-CNN service with the assets in this folder.

## Repository Structure

- `setup_server.sh`: Automated server configuration and launch script.
- `FastTime/` and `third_party/`: Supporting libraries and accelerated implementations.
- `dataset/`: Time-series datasets required by the service.
- `ssa_cnn_server_optimized.py`: Entry point for serving the SSA-CNN model.

Make changes as needed for your infrastructure, then rerun the setup script to apply them.
