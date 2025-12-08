# 实验运行完整指南

## 环境准备

### 1. Python 环境
确保 Python 3.8+ 已安装：
```bash
python --version
```

### 2. 安装依赖
```bash
pip install -r requirements.txt
```

### 3. 安装 tsdecomp
tsdecomp 是自定义的时间序列分解包，需要单独安装：
```bash
# 方法 1: 如果有 setup.py
cd /path/to/tsdecomp
pip install -e .

# 方法 2: 添加到 PYTHONPATH
export PYTHONPATH="/path/to/tsdecomp:$PYTHONPATH"
```

验证安装：
```bash
python -c "import tsdecomp; print('tsdecomp installed successfully')"
```

## 快速测试

在运行完整实验前，先测试环境配置：

```bash
cd server_ready
python scripts/test_experiment_setup.py
```

预期输出：
```
✅ Baseline test passed!
✅ Decomposition test passed!
✅ All tests passed!
```

## 运行实验

### 方式 1: 批量运行所有实验（推荐）

**Linux/Mac 服务器**：
```bash
# 使用 nohup 后台运行
nohup bash scripts/run_all_ett_exchange_experiments.sh > experiment_log.txt 2>&1 &

# 获取进程 ID
echo $!

# 实时监控进度
tail -f experiment_log.txt

# 如需停止
kill <进程ID>
```

**Windows**：
```powershell
# 前台运行（推荐，可以看到进度）
powershell -ExecutionPolicy Bypass -File scripts\run_all_ett_exchange_experiments.ps1

# 或后台运行
Start-Process powershell -ArgumentList "-ExecutionPolicy Bypass -File scripts\run_all_ett_exchange_experiments.ps1" -WindowStyle Hidden
```

### 方式 2: 逐个数据集运行

```bash
# ETTh1
python -m features.decomp_linear_bench.cli run --config configs/decomp_linear/etth1_baseline.yaml
python -m features.decomp_linear_bench.cli run --config configs/decomp_linear/etth1_all.yaml

# ETTh2
python -m features.decomp_linear_bench.cli run --config configs/decomp_linear/etth2_baseline.yaml
python -m features.decomp_linear_bench.cli run --config configs/decomp_linear/etth2_all.yaml

# ETTm1
python -m features.decomp_linear_bench.cli run --config configs/decomp_linear/ettm1_baseline.yaml
python -m features.decomp_linear_bench.cli run --config configs/decomp_linear/ettm1_all.yaml

# ETTm2
python -m features.decomp_linear_bench.cli run --config configs/decomp_linear/ettm2_baseline.yaml
python -m features.decomp_linear_bench.cli run --config configs/decomp_linear/ettm2_all.yaml

# Exchange Rate
python -m features.decomp_linear_bench.cli run --config configs/decomp_linear/exchange_baseline.yaml
python -m features.decomp_linear_bench.cli run --config configs/decomp_linear/exchange_all.yaml
```

## 监控进度

### 查看日志
```bash
# 实时查看
tail -f experiment_log.txt

# 查看最近 50 行
tail -n 50 experiment_log.txt

# 搜索错误
grep -i error experiment_log.txt
```

### 检查输出文件
```bash
# 查看已完成的实验
ls outputs/decomp_linear_bench/baseline/
ls outputs/decomp_linear_bench/all_methods/

# 统计完成数量
find outputs -name "metrics_summary_*.csv" | wc -l
```

## 预期运行时间

| 数据集 | Baseline | All Methods | 小计 |
|--------|----------|-------------|------|
| ETTh1 | 2分钟 | 20-30分钟 | ~32分钟 |
| ETTh2 | 2分钟 | 20-30分钟 | ~32分钟 |
| ETTm1 | 5分钟 | 40-60分钟 | ~65分钟 |
| ETTm2 | 5分钟 | 40-60分钟 | ~65分钟 |
| Exchange | 1分钟 | 10-15分钟 | ~16分钟 |
| **总计** | 15分钟 | 3-4小时 | **~4小时** |

## 加速运行

如果 4 小时太长，可以修改配置文件：

### 方案 1: 减少分解方法
编辑 `configs/decomp_linear/*_all.yaml`：
```yaml
decomp:
  - method: STL
    params: {period: 24}
  - method: MA_BASELINE
    params: {window: 24}
  - method: STD_MULTI
    params: {block_size: 24}
  # 注释掉 SSA, EMD, WAVELET
```

### 方案 2: 减少预测长度
```yaml
horizons: [96, 336]  # 原来是 [96, 192, 336, 720]
```

### 方案 3: 组合使用
同时使用方案 1 和 2，可以减少到约 1 小时。

## 结果分析

实验完成后，运行分析脚本：

```bash
python scripts/analyze_results.py
```

生成的文件：
- `outputs/decomp_linear_bench/analysis/RESULTS_SUMMARY.md` - Markdown 报告
- `outputs/decomp_linear_bench/analysis/summary_by_method_ablation.csv` - 方法排名
- `outputs/decomp_linear_bench/analysis/best_method_per_dataset_horizon.csv` - 最佳配置
- `outputs/decomp_linear_bench/analysis/component_comparison.csv` - 成分对比
- `outputs/decomp_linear_bench/analysis/full_comparison.csv` - 完整对比表

## 故障排查

### 问题 1: ImportError: No module named 'tsdecomp'
**解决**：
```bash
# 检查 tsdecomp 是否安装
python -c "import tsdecomp"

# 如果未安装，需要单独安装该包
# 联系开发者或查看 tsdecomp 安装说明
```

### 问题 2: FileNotFoundError: dataset/ETTh1.csv
**解决**：
```bash
# 确认当前目录
pwd  # 应该在 server_ready/ 目录

# 确认数据文件存在
ls dataset/*.csv

# 如果数据文件缺失，从原始数据源下载
```

### 问题 3: 分解方法失败
**症状**：看到 "Warning: Decomposition failed"
**解决**：
- 检查数据是否包含 NaN 或 Inf
- 某些方法对参数敏感（如 STL 的 period）
- 可以在配置文件中注释掉有问题的方法

### 问题 4: 内存不足
**解决**：
```yaml
# 编辑配置文件，减少 horizons
horizons: [96]  # 只测试最短的预测长度
```

### 问题 5: 运行太慢
**解决**：
- 参考"加速运行"部分
- 使用多进程并行运行不同数据集
- 在配置文件中去掉 EMD（最慢的方法）

## 验证结果

### 检查完整性
```bash
# 应该有 10 个结果文件（5 数据集 × 2 类型）
find outputs/decomp_linear_bench -name "metrics_summary_*.csv" | wc -l
```

### 检查数据质量
```bash
# 查看某个结果文件
head outputs/decomp_linear_bench/baseline/ETTh1/metrics_summary_by_method_horizon_ablation.csv

# 确认没有全是 NaN 的列
```

## 并行运行（高级）

如果有多核 CPU，可以并行运行不同数据集：

```bash
# 终端 1
python -m features.decomp_linear_bench.cli run --config configs/decomp_linear/etth1_all.yaml &

# 终端 2
python -m features.decomp_linear_bench.cli run --config configs/decomp_linear/etth2_all.yaml &

# 终端 3
python -m features.decomp_linear_bench.cli run --config configs/decomp_linear/ettm1_all.yaml &

# 等待所有任务完成
wait
```

## 完成后的清理

```bash
# 压缩结果文件
tar -czf results_$(date +%Y%m%d).tar.gz outputs/

# 备份日志
cp experiment_log.txt experiment_log_$(date +%Y%m%d).txt

# 清理临时文件（如果有）
rm -rf __pycache__
```

## 下载结果

从服务器下载结果到本地：

```bash
# 使用 scp
scp -r user@server:/path/to/server_ready/outputs ./

# 或使用 rsync
rsync -avz user@server:/path/to/server_ready/outputs ./
```

## 技术支持

如遇到问题：
1. 查看 `experiment_log.txt` 中的错误信息
2. 运行 `python scripts/test_experiment_setup.py` 诊断环境
3. 检查 `outputs/` 目录中的部分结果
4. 参考原仓库的 `DATA_LEAKAGE_FIX.md` 等文档
