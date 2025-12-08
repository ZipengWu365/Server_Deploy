# Server Ready 文件夹清单

## ✅ 已包含的核心文件

### 代码文件
- [x] `features/decomp_linear_bench/` - 完整的实验代码
  - [x] `cli.py` - 命令行接口
  - [x] `runner.py` - 实验运行器（已修复数据泄露）
  - [x] `builder.py` - 特征构建
  - [x] `ablations.py` - 成分选择
  - [x] `learners.py` - 学习器
  - [x] `configs.py` - 配置管理
  - [x] `report.py` - 报告生成

### 配置文件
- [x] `configs/decomp_linear/` - 所有实验配置
  - [x] `EXPERIMENT_DESIGN.md` - 实验设计文档
  - [x] `etth1_baseline.yaml` & `etth1_all.yaml`
  - [x] `etth2_baseline.yaml` & `etth2_all.yaml`
  - [x] `ettm1_baseline.yaml` & `ettm1_all.yaml`
  - [x] `ettm2_baseline.yaml` & `ettm2_all.yaml`
  - [x] `exchange_baseline.yaml` & `exchange_all.yaml`
  - [x] 其他配置文件（electricity, traffic, weather 等）

### 脚本文件
- [x] `scripts/run_all_ett_exchange_experiments.sh` - Linux/Mac 批量运行
- [x] `scripts/run_all_ett_exchange_experiments.ps1` - Windows 批量运行
- [x] `scripts/test_experiment_setup.py` - 快速测试
- [x] `scripts/analyze_results.py` - 结果分析

### 数据集
- [x] `dataset/ETTh1.csv`
- [x] `dataset/ETTh2.csv`
- [x] `dataset/ETTm1.csv`
- [x] `dataset/ETTm2.csv`
- [x] `dataset/exchange_rate.csv`
- [x] `dataset/` 还包含其他数据集（electricity, traffic, weather, ILI）

### 文档
- [x] `README.md` - 项目说明
- [x] `RUN_EXPERIMENTS.md` - 详细运行指南
- [x] `DEPLOY.md` - 部署说明
- [x] `requirements.txt` - Python 依赖
- [x] `.gitignore` - Git 忽略文件

## ❌ 不包含的文件（保留在主项目中）

### 开发文件
- 原项目的 `archive/` 文件夹
- 其他实验脚本（非 ETT/Exchange 的）
- 开发工具和临时文件

### 环境文件
- `ssacnn_env/` - Python 虚拟环境（需在服务器上重新创建）
- `third_party/` - 第三方库（通过 pip 安装）
- `FastTime/` - 其他项目文件

### 输出和日志
- `outputs/` - 实验输出（运行后生成）
- `__pycache__/` - Python 缓存（自动生成）

## 📦 文件大小估计

| 类别 | 估计大小 |
|------|----------|
| 代码文件 | ~100 KB |
| 配置文件 | ~50 KB |
| 文档 | ~50 KB |
| 数据集 | ~15-20 MB |
| **总计** | **~20 MB** |

## 🚀 上传到 GitHub 前的检查

```bash
cd server_ready

# 1. 确认所有核心文件存在
ls features/decomp_linear_bench/
ls configs/decomp_linear/
ls scripts/
ls dataset/

# 2. 确认没有敏感信息
grep -r "password\|secret\|token" . 2>/dev/null

# 3. 测试代码可以运行
python scripts/test_experiment_setup.py

# 4. 检查文件大小
du -sh .

# 5. 初始化 Git
git init
git add .
git status
```

## ⚠️ 注意事项

### 需要单独处理
1. **tsdecomp 包** - 需要单独安装或上传
2. **虚拟环境** - 在服务器上重新创建
3. **outputs/ 文件夹** - 实验运行后自动生成

### 可选的优化
1. 如果数据集很大，可以：
   - 从 `.gitignore` 中排除 `dataset/*.csv`
   - 在 README 中添加数据下载链接
   - 使用 Git LFS 管理大文件

2. 如果只运行特定数据集，可以：
   - 只保留需要的配置文件
   - 删除不需要的数据集

## 📋 部署检查清单

在服务器上克隆后：

- [ ] 创建虚拟环境：`python3 -m venv venv`
- [ ] 激活环境：`source venv/bin/activate`
- [ ] 安装依赖：`pip install -r requirements.txt`
- [ ] 安装 tsdecomp：需要单独处理
- [ ] 测试环境：`python scripts/test_experiment_setup.py`
- [ ] 运行实验：`bash scripts/run_all_ett_exchange_experiments.sh`
- [ ] 监控进度：`tail -f experiment_log.txt`
- [ ] 分析结果：`python scripts/analyze_results.py`

## 🎯 准备完成

`server_ready/` 文件夹现在包含了在服务器上运行实验所需的所有文件。可以直接：
1. 上传到 GitHub
2. 在服务器上克隆
3. 按照 `DEPLOY.md` 的说明运行

**祝实验顺利！🚀**
