# 🎯 快速参考卡

## 一行命令启动

### Linux/Mac
```bash
bash setup.sh && nohup bash scripts/run_all_ett_exchange_experiments.sh > experiment_log.txt 2>&1 &
```

### Windows
```powershell
.\setup.ps1; powershell -ExecutionPolicy Bypass -File scripts\run_all_ett_exchange_experiments.ps1
```

## 常用命令

### 测试环境
```bash
python scripts/test_experiment_setup.py
```

### 运行实验
```bash
# 前台运行
bash scripts/run_all_ett_exchange_experiments.sh

# 后台运行
nohup bash scripts/run_all_ett_exchange_experiments.sh > experiment_log.txt 2>&1 &
```

### 查看进度
```bash
tail -f experiment_log.txt
```

### 分析结果
```bash
python scripts/analyze_results.py
```

### 停止实验
```bash
# 查找进程
ps aux | grep python

# 停止进程
kill <PID>
```

## 文件位置

| 内容 | 位置 |
|------|------|
| 配置 | `configs/decomp_linear/*.yaml` |
| 数据 | `dataset/*.csv` |
| 代码 | `features/decomp_linear_bench/*.py` |
| 脚本 | `scripts/*.sh` 或 `*.ps1` |
| 结果 | `outputs/decomp_linear_bench/` |
| 日志 | `experiment_log.txt` |

## 关键数字

- **数据集**: 5 个（ETTh1, ETTh2, ETTm1, ETTm2, Exchange）
- **分解方法**: 6 个（STL, SSA, EMD, MA, Wavelet, STD）
- **成分**: 2 个（+T 趋势, +S 季节性）
- **预测长度**: 4 个（96, 192, 336, 720）
- **总实验**: ~245 个配置
- **运行时间**: ~4 小时

## 加速技巧

编辑配置文件减少实验量：

```yaml
# configs/decomp_linear/*_all.yaml

# 只保留 3 个方法
decomp:
  - method: STL
    params: {period: 24}
  - method: MA_BASELINE
    params: {window: 24}
  - method: STD_MULTI
    params: {block_size: 24}

# 只测试 2 个长度
horizons: [96, 336]
```

**节省时间**: 从 4 小时 → 1 小时

## 故障排查

| 问题 | 解决方案 |
|------|----------|
| 找不到 tsdecomp | `export PYTHONPATH="/path/to/tsdecomp:$PYTHONPATH"` |
| 找不到数据文件 | 确认在 `server_ready/` 目录 |
| 内存不足 | 减少 horizons: `[96]` |
| 运行太慢 | 注释掉 EMD 方法 |

## 检查进度

```bash
# 查看完成的实验数
find outputs -name "*.csv" | wc -l

# 应该有 10 个文件（5数据集 × 2类型）
ls outputs/decomp_linear_bench/baseline/
ls outputs/decomp_linear_bench/all_methods/
```

## 帮助文档

- 快速开始: `README.md`
- 部署指南: `DEPLOY.md`
- 运行指南: `RUN_EXPERIMENTS.md`
- 检查清单: `CHECKLIST.md`
