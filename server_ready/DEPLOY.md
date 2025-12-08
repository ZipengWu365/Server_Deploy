# 🚀 服务器一键部署指南

## 第一步：上传到 GitHub

```bash
cd server_ready
git init
git add .
git commit -m "Initial commit: Ridge + Decomposition experiments"
git branch -M main
git remote add origin https://github.com/YourUsername/decomp-linear-bench.git
git push -u origin main
```

## 第二步：在服务器上克隆

```bash
# SSH 登录到服务器
ssh user@your-server.com

# 克隆仓库
git clone https://github.com/YourUsername/decomp-linear-bench.git
cd decomp-linear-bench
```

## 第三步：安装依赖

```bash
# 创建虚拟环境（推荐）
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
.\venv\Scripts\Activate.ps1  # Windows

# 安装依赖
pip install -r requirements.txt

# 安装 tsdecomp（需要单独处理）
# 如果 tsdecomp 在其他仓库：
# git clone https://github.com/YourUsername/tsdecomp.git
# cd tsdecomp
# pip install -e .
# cd ..
```

## 第四步：测试环境

```bash
python scripts/test_experiment_setup.py
```

看到 `✅ All tests passed!` 表示环境正确。

## 第五步：运行实验

```bash
# 后台运行所有实验
nohup bash scripts/run_all_ett_exchange_experiments.sh > experiment_log.txt 2>&1 &

# 记录进程 ID
echo $! > experiment_pid.txt

# 监控进度
tail -f experiment_log.txt
```

## 第六步：分析结果

```bash
# 等待实验完成后（约 4 小时）
python scripts/analyze_results.py

# 查看报告
cat outputs/decomp_linear_bench/analysis/RESULTS_SUMMARY.md
```

## 第七步：下载结果

```bash
# 在本地机器上运行
scp -r user@server:/path/to/decomp-linear-bench/outputs ./
```

## 快捷命令

```bash
# 一键运行（测试 + 实验）
python scripts/test_experiment_setup.py && \
nohup bash scripts/run_all_ett_exchange_experiments.sh > experiment_log.txt 2>&1 &

# 检查实验状态
ps aux | grep python

# 停止实验
kill $(cat experiment_pid.txt)

# 清理重新运行
rm -rf outputs/
git pull
```

## 注意事项

1. **确保数据集文件存在**：`dataset/` 目录下应有 5 个 CSV 文件
2. **tsdecomp 包必须安装**：这是自定义包，需要单独处理
3. **预计运行时间**：约 4 小时，确保服务器稳定
4. **磁盘空间**：确保至少有 1GB 可用空间存储结果
5. **内存要求**：建议至少 4GB RAM

## 故障排查

### 找不到 tsdecomp
```bash
# 临时解决：添加到 PYTHONPATH
export PYTHONPATH="/path/to/tsdecomp:$PYTHONPATH"
echo 'export PYTHONPATH="/path/to/tsdecomp:$PYTHONPATH"' >> ~/.bashrc
```

### 实验中断
```bash
# 查看日志找出问题
tail -100 experiment_log.txt

# 重新运行失败的数据集
python -m features.decomp_linear_bench.cli run --config configs/decomp_linear/etth1_all.yaml
```

### 加速运行
编辑配置文件减少实验量，详见 `RUN_EXPERIMENTS.md`

---

**完成后别忘了备份结果！**
```bash
tar -czf results_$(date +%Y%m%d_%H%M%S).tar.gz outputs/
```
