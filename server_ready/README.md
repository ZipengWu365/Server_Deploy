# Ridge 回归 + 时间序列分解成分实验

这是一个干净的服务器部署包，包含运行实验所需的所有文件。

## 📦 目录结构

```
server_ready/
├── features/
│   └── decomp_linear_bench/     # 实验核心代码
│       ├── __init__.py
│       ├── cli.py               # 命令行接口
│       ├── runner.py            # 实验运行器（已修复数据泄露）
│       ├── builder.py           # 特征构建
│       ├── ablations.py         # 成分选择
│       ├── learners.py
│       ├── configs.py
│       └── report.py
│
├── configs/
│   └── decomp_linear/           # 实验配置
│       ├── EXPERIMENT_DESIGN.md
│       ├── etth1_baseline.yaml
│       ├── etth1_all.yaml
│       ├── etth2_baseline.yaml
│       ├── etth2_all.yaml
│       ├── ettm1_baseline.yaml
│       ├── ettm1_all.yaml
│       ├── ettm2_baseline.yaml
│       ├── ettm2_all.yaml
│       ├── exchange_baseline.yaml
│       └── exchange_all.yaml
│
├── scripts/
│   ├── run_all_ett_exchange_experiments.sh   # Linux/Mac 批量运行
│   ├── run_all_ett_exchange_experiments.ps1  # Windows 批量运行
│   ├── test_experiment_setup.py              # 快速测试
│   └── analyze_results.py                    # 结果分析
│
├── dataset/                     # 数据集（CSV 文件）
│   ├── ETTh1.csv               # ✅ 包含
│   ├── ETTh2.csv               # ✅ 包含
│   ├── ETTm1.csv               # ✅ 包含
│   ├── ETTm2.csv               # ✅ 包含
│   ├── exchange_rate.csv       # ✅ 包含
│   ├── weather.csv             # ✅ 包含
│   ├── ILI.csv                 # ✅ 包含
│   └── national_illness.csv    # ✅ 包含
│   # 注意: electricity.csv 和 traffic.csv 因文件过大未包含
│
├── README.md                    # 本文件
├── requirements.txt             # Python 依赖
└── RUN_EXPERIMENTS.md          # 详细运行指南
```

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 测试环境

```bash
python scripts/test_experiment_setup.py
```

### 3. 运行实验

**Linux/Mac 服务器**：
```bash
# 后台运行
nohup bash scripts/run_all_ett_exchange_experiments.sh > experiment_log.txt 2>&1 &

# 查看进度
tail -f experiment_log.txt
```

**Windows**：
```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_all_ett_exchange_experiments.ps1
```

### 4. 分析结果

```bash
python scripts/analyze_results.py
```

## 📊 实验内容

- **5 个数据集**: ETTh1, ETTh2, ETTm1, ETTm2, Exchange Rate
- **6 种分解方法**: STL, SSA, EMD, MA_BASELINE, WAVELET, STD_MULTI
- **2 种成分**: +T (趋势), +S (季节性)
- **4 个预测长度**: 96, 192, 336, 720
- **预计运行时间**: 约 4 小时

## ⚠️ 重要说明

### 数据泄露已修复
代码已修复数据泄露问题，确保：
- 每个预测窗口独立分解
- 只使用 lookback 窗口内的数据
- 严格模拟真实预测场景

详见原仓库的 `DATA_LEAKAGE_FIX.md`

### 加速运行
如果 4 小时太长，可以编辑配置文件减少实验量：

```yaml
# configs/decomp_linear/*_all.yaml

# 减少分解方法
decomp:
  - method: STL
    params: {period: 24}
  - method: STD_MULTI
    params: {block_size: 24}

# 减少预测长度
horizons: [96, 336]
```

这样可以减少到约 1-2 小时。

## 📈 输出结果

实验完成后，结果保存在：
- `outputs/decomp_linear_bench/baseline/` - Baseline 结果
- `outputs/decomp_linear_bench/all_methods/` - 全方法结果
- `outputs/decomp_linear_bench/analysis/` - 分析报告（运行 analyze_results.py 后）

## 🔧 依赖要求

- Python 3.8+
- numpy
- pandas
- scikit-learn
- statsmodels
- PyEMD
- PyWavelets
- pyyaml
- tsdecomp (自定义包，需确保已安装)

## 📝 详细文档

更多详细信息，请参考：
- `RUN_EXPERIMENTS.md` - 完整运行指南
- `configs/decomp_linear/EXPERIMENT_DESIGN.md` - 实验设计文档

## 🐛 故障排查

### 问题：找不到模块
```bash
# 确保在项目根目录
cd server_ready
python -m features.decomp_linear_bench.cli run --config configs/decomp_linear/etth1_baseline.yaml
```

### 问题：找不到数据文件
```bash
# 确认数据集存在
ls dataset/*.csv
```

### 问题：tsdecomp 未安装
```bash
# 需要单独安装 tsdecomp 包
# 或联系开发者获取安装说明
```

## 📧 联系方式

如有问题，请查看原仓库或联系开发者。
