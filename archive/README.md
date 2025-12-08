# 归档说明 / Archive README

本目录存档了与新的**多尺度分解实验**无关的旧代码和结果文件。

Created: 2025-12-02

## 归档内容 / Archived Content

### 1. legacy_experiments/
旧的实验脚本，主要是基于单尺度的 CNN/ESN/SSA 实验：
- `1Dcnn_server_optimized.py` / `1Dcnn_server_optimized2.py` - 旧的CNN优化版本
- `ESN_server_optimized.py` - Echo State Network 实验
- `ssa_cnn_*.py` / `ssa_mamba.py` - 旧的SSA+深度学习混合模型
- `run_all_experiments.py` / `run_component_comparison.py` - 旧的批量运行脚本

### 2. legacy_analysis/
旧的分析脚本：
- `analyze_*.py` - 各种旧的结果分析工具
- `verify_fair_comparison.py` - 旧的对比验证脚本
- `TRAIN_TEST_SPLIT_ISSUE.py` - 数据划分问题调试

### 3. legacy_configs/
旧的YAML配置文件：
- `quick_std.yaml` - 单尺度STD快速测试
- `compare_components.yaml` - 旧的组件对比
- `etth1_std_vs_others.yaml` - 旧的方法对比

### 4. legacy_results/
历史实验结果：
- `*_cnn_*.csv` - CNN相关实验结果
- `*_linear_*.csv` - 线性模型实验结果
- `std_component_ablation_*.csv` - 旧的STD消融实验结果
- `results_*.txt` - 文本格式的结果摘要

### 5. temp_tests/
临时测试文件：
- `test_multiscale.py` - 多尺度特征提取的快速验证脚本

## 当前活跃文件 / Active Files

**实验模块**:
- `features/decomp_linear_bench/` - 新的多尺度线性基准模块
- `third_party/tsdecomp/` - 统一的分解方法包

**配置文件** (`configs/decomp_linear/`):
- `quick_multiscale.yaml` ✅
- `compare_multiscale_methods.yaml` ⏳ (正在运行)
- `scale_ablation.yaml`
- `etth1_all.yaml`, `etth2_all.yaml`, `ettm1_all.yaml`, `ettm2_all.yaml`, `exchange_all.yaml`
- `etth1_baseline.yaml`, `etth2_baseline.yaml`

**核心代码**:
- `std_ablation_study.py` - STD消融研究（已更新支持多尺度）
- `decomp_methods.py` - 分解方法实现（部分已迁移到tsdecomp）
- `fasttimes.py` - FastTimes分解核心

**脚本** (`scripts/`):
- `run_all_experiments.ps1` - 新的批量运行脚本
- `train_*.sh` - 训练脚本

## 恢复方法 / Recovery

如需恢复归档的文件：
```powershell
# 恢复到原位置
Move-Item -Path "archive/legacy_experiments/*" -Destination "." -Force
```

## 注意事项 / Notes

1. **不建议删除** - 这些文件仅归档，保留以备查阅
2. **可安全忽略** - 多尺度实验不需要这些文件
3. **独立运行** - 如需运行归档的脚本，需先恢复并检查依赖
