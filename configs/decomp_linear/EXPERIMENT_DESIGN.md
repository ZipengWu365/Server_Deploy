# Ridge 回归 + 分解成分预测实验设计

## 实验目标

测试 **Ridge 回归 + 时间序列分解成分** 能否提高预测性能。

**核心问题**：哪种分解方法提取的成分（趋势 Trend / 季节性 Seasonal）能最有效地增强 Ridge 回归的长期预测能力？

**Baseline**：Ridge 回归 + 原始序列，lookback window = 336

---

## 实验设计概览

### 核心思路
1. **Baseline**: Ridge 回归直接使用原始序列的 lookback window (336) 作为特征
2. **实验组**: Ridge 回归使用 **lookback + 分解成分** 作为特征
   - lookback + 趋势 (+T)
   - lookback + 季节性 (+S)  
3. **多尺度分解**: 每种方法在多个时间尺度上提取成分
4. **对比**: 添加各成分后 vs Baseline 的预测性能差异

---

## 数据集配置

### 数据集划分（Train / Val / Test）

| 数据集 | Train | Val | Test | 频率 |
|--------|-------|-----|------|------|
| ETTh1 | 8,545 | 2,881 | 2,881 | 小时 |
| ETTh2 | 8,545 | 2,881 | 2,881 | 小时 |
| ETTm1 | 34,465 | 11,521 | 11,521 | 15分钟 |
| ETTm2 | 34,465 | 11,521 | 11,521 | 15分钟 |
| Electricity | 18,317 | 2,633 | 5,261 | 小时 |
| Traffic | 12,185 | 1,757 | 3,509 | 小时 |
| Weather | 36,792 | 5,271 | 10,540 | 10分钟 |
| Exchange Rate | 5,120 | 665 | 1,422 | 日 |

### 各数据集的分解参数配置

基于数据集的采样频率选择合适的周期/窗口参数：

| 数据集 | 频率 | 日周期 | 周周期 | 推荐 period/window |
|--------|------|--------|--------|------------------------|
| ETTh1/h2 | 小时 | 24 | 168 | 24 (日) 或 168 (周) |
| ETTm1/m2 | 15分钟 | 96 | 672 | 96 (日) 或 672 (周) |
| Electricity | 小时 | 24 | 168 | 24 (日) |
| Traffic | 小时 | 24 | 168 | 24 (日) |
| Weather | 10分钟 | 144 | 1008 | 144 (日) |
| Exchange Rate | 日 | - | 5 | 5 (周) 或 21 (月) |

---

## 分解方法配置 (tsdecomp)

### 分解工具

所有分解方法均通过 **`tsdecomp`** 包的统一 API 调用：

```python
from tsdecomp import decompose, DecompositionConfig

# 示例：STL 分解
cfg = DecompositionConfig(method="STL", params={"period": 24})
result = decompose(x, cfg)
# result.trend, result.season, result.residual
```

### tsdecomp 注册方法及参数

| 方法 | 注册名 | 必需参数 | 可选参数 | 各数据集推荐配置 |
|------|--------|----------|----------|------------------|
| **STL** | `"STL"` | `period` | - | ETTh: 24, ETTm: 96, Weather: 144, Exchange: 5 |
| **MSTL** | `"MSTL"` | `periods` (list) | - | ETTh: [24,168], ETTm: [96,672] |
| **SSA** | `"SSA"` | - | `window`, `n_components` | window=48~100, n_components=10 |
| **EMD** | `"EMD"` | - | `max_imfs` | 默认即可 |
| **MA_BASELINE** | `"MA_BASELINE"` | - | `window` | window=24~168 |
| **WAVELET** | `"WAVELET"` | - | `wavelet`, `level` | wavelet="db4", level=3~5 |
| **STD_MULTI** | `"STD_MULTI"` | - | `block_size` | ETTh: 24, ETTm: 96 |

### 成分定义

- **T (Trend)**: 趋势成分 - 长期变化趋势
- **S (Seasonal)**: 季节性成分 - 周期性模式  

---

## 实验配置

### 实验参数

| 参数 | 值 | 说明 |
|------|-----|------|
| Lookback Window | 336 | 输入序列长度 |
| Horizons | `[96, 192, 336, 720]` | 预测长度 |
| Model | Ridge Regression | 线性回归 + L2正则化 |
| Alpha | 1.0 | 正则化强度 |
| Seed | 42 | 随机种子 |

### 成分实验矩阵

每个实验测试 **lookback + 分解成分** 对 Ridge 回归的影响：

**特征构造方式**：
```python
# 输入特征 = 原始序列 拼接 分解成分
feature = np.concatenate([raw[-336:], component[-336:]])  # shape: (672,)
```

| 方法 | 成分 | 输入特征 | 特征维度 |
|------|------|----------|----------|
| Baseline | - | `raw[-336:]` | 336 |
| STL | +T | `concat(raw[-336:], trend[-336:])` | 672 |
| STL | +S | `concat(raw[-336:], seasonal[-336:])` | 672 |
| SSA | +T | `concat(raw[-336:], trend[-336:])` | 672 |
| SSA | +S | `concat(raw[-336:], seasonal[-336:])` | 672 |
| ... | ... | ... | ... |

### 评估指标

| 指标 | 说明 |
|------|------|
| **MAE** | Mean Absolute Error (主要指标) |
| **MSE** | Mean Squared Error (辅助指标) |

---

## 实验执行任务列表

### Phase 1: 环境准备
- [ ] 验证 `tsdecomp` 包安装完成
- [ ] 准备所有数据集文件
- [ ] 验证数据集划分比例正确

### Phase 2: Baseline 实验
- [ ] 运行 Ridge Baseline（原始序列，lookback=336）
- [ ] 记录所有数据集、所有 horizon 的 Baseline 性能

### Phase 3: 分解成分实验
- [ ] **STL 分解**: 所有数据集 × T, S 成分
- [ ] **SSA 分解**: 所有数据集 × T, S 成分
- [ ] **EMD 分解**: 所有数据集 × T, S 成分
- [ ] **MA_BASELINE 分解**: 所有数据集 × T, S 成分
- [ ] **WAVELET 分解**: 所有数据集 × T, S 成分
- [ ] **STD_MULTI 分解**: 所有数据集 × T, S 成分

### Phase 4: 结果分析
- [ ] 汇总所有实验结果到统一 CSV
- [ ] 计算各成分相对 Baseline 的提升/下降
- [ ] 生成可视化报告

---

## 预期成果

### 成分 vs Baseline 对比表 (示例)

| Dataset | Method | Component | H | MAE | MSE | vs Baseline |
|---------|--------|-----------|---|-----|-----|-------------|
| ETTh1 | Baseline | Raw | 96 | 0.37 | 0.30 | - |
| ETTh1 | STL | +T | 96 | 0.35 | 0.28 | **-5.4%** |
| ETTh1 | STL | +S | 96 | 0.38 | 0.31 | +2.7% |
| ETTh1 | SSA | +T | 96 | 0.34 | 0.27 | **-8.1%** |
| ... | ... | ... | ... | ... | ... | ... |

### 研究假设（待验证）
- **假设1**: 趋势成分 (T) 在大多数方法/数据集上优于 Baseline
- **假设2**: 季节性成分 (S) 在周期性强的数据（如 Electricity、Traffic）上表现更好
- **假设3**: SSA 和 Wavelet 的趋势成分优于 STL
- **假设4**: 长期预测 (720) 更依赖趋势成分，短期预测 (96) 更依赖季节性成分

---

## 文件结构

```
configs/decomp_linear/
├── EXPERIMENT_DESIGN.md          # 本文件
├── etth1_baseline.yaml           # ETTh1 Baseline
├── etth2_baseline.yaml           # ETTh2 Baseline
├── ettm1_baseline.yaml           # ETTm1 Baseline
├── ettm2_baseline.yaml           # ETTm2 Baseline
├── exchange_baseline.yaml        # Exchange Rate Baseline
├── etth1_all.yaml                # ETTh1 全方法对比
├── etth2_all.yaml                # ETTh2 全方法对比
├── ettm1_all.yaml                # ETTm1 全方法
├── ettm2_all.yaml                # ETTm2 全方法
├── electricity_all.yaml          # Electricity 全方法
├── traffic_all.yaml              # Traffic 全方法
├── weather_all.yaml              # Weather 全方法
└── exchange_all.yaml             # Exchange Rate 全方法
```
