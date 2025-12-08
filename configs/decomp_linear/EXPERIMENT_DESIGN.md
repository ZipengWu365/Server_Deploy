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

### 分解模式选择：Per-Window vs Global Pattern

**重要决策**：测试时是否需要重新分解？

#### 模式A：Per-Window Decomposition（当前实现）
```python
# 每个滑动窗口独立分解
for window in test_windows:
    trend, seasonal = decompose(window)
    feature = concat(window, trend, seasonal)
```
- ✅ 适用于：EMD, Wavelet（自适应方法）
- ✅ 捕捉局部变化
- ❌ 计算量大，假设测试时可以看到完整lookback窗口

#### 模式B：Global Pattern Decomposition（推荐用于STL/SSA）
```python
# 只在训练集上分解一次
train_trend, train_seasonal = decompose(train_data)

# 测试时使用全局模式
for i, window in enumerate(test_windows):
    # 使用训练集学到的周期/趋势模式
    trend_slice = train_trend[i:i+336]
    seasonal_slice = train_seasonal[i:i+336]
    feature = concat(window, trend_slice, seasonal_slice)
```
- ✅ 适用于：STL, SSA（周期性稳定的方法）
- ✅ 真实场景：周期模式在训练时确定
- ✅ 计算高效
- ❌ 假设模式固定

**本实验采用模式A**，未来可扩展支持模式B。

---

### 多尺度分解实验设计

每种分解方法在**多个时间尺度**上提取成分：

#### 单尺度实验（基础）
| 方法 | 尺度参数 | 成分 | 输入特征 | 特征维度 |
|------|----------|------|----------|----------|
| Baseline | - | - | `raw[-336:]` | 336 |
| STL-24 | period=24 | +T | `concat(raw, trend_24)` | 672 |
| STL-24 | period=24 | +S | `concat(raw, seasonal_24)` | 672 |
| STL-168 | period=168 | +T | `concat(raw, trend_168)` | 672 |
| STL-168 | period=168 | +S | `concat(raw, seasonal_168)` | 672 |

#### 多尺度实验（组合）
```python
# 方案1: 多个尺度的趋势成分
feature = concat(raw, trend_24, trend_168)  # dim: 336+336+336=1008

# 方案2: 多个尺度的季节性成分  
feature = concat(raw, seasonal_24, seasonal_168)

# 方案3: 混合多尺度
feature = concat(raw, trend_24, seasonal_168)
```

| 实验 | 成分组合 | 输入特征 | 特征维度 |
|------|----------|----------|----------|
| Multi-T | 多尺度趋势 | `concat(raw, T_24, T_168)` | 1008 |
| Multi-S | 多尺度季节 | `concat(raw, S_24, S_168)` | 1008 |
| Multi-TS | 单尺度T+S | `concat(raw, T_24, S_24)` | 1008 |
| Multi-Mix | 跨尺度混合 | `concat(raw, T_24, S_168)` | 1008 |

---

### SSA 特定配置

SSA 方法的参数需要根据数据集调整：

#### SSA 窗口大小（Window Size）
```python
# 经验法则：window = lookback / 2 到 lookback / 3
```

| 数据集 | Lookback | 推荐 Window | 说明 |
|--------|----------|-------------|------|
| ETTh1/h2 | 336 | 112 ~ 168 | ~1周数据 |
| ETTm1/m2 | 336 | 112 ~ 168 | ~1.17天 |
| Exchange | 336 | 112 ~ 168 | ~16周 |

#### SSA 成分分组（Trend vs Seasonal）
```python
# SSA 分解后得到多个特征值对应的成分
# 需要手动分组为 Trend 和 Seasonal

# 方案1: 前N个为趋势，其余为季节
n_components = 10
trend_indices = [0, 1, 2]  # 前3个主成分为趋势
seasonal_indices = [3, 4, 5, 6]  # 中间若干为季节

# 方案2: 根据周期性判断
# - 低频成分 (前1-3) → Trend
# - 中频成分 (4-8) → Seasonal  
# - 高频成分 (9+) → Noise (舍弃)
```

**推荐 SSA 配置**：
```yaml
SSA:
  window: 112  # lookback / 3
  n_components: 10
  trend_indices: [0, 1, 2]  # 前3个特征值
  seasonal_indices: [3, 4, 5, 6, 7]  # 中间5个
```

---

### 完整实验矩阵

| 方法 | 尺度 | 成分 | 输入特征 | 特征维度 |
|------|------|------|----------|----------|
| **Baseline** | - | RAW | `raw` | 336 |
| **STL** | 24 | +T | `concat(raw, T_24)` | 672 |
| **STL** | 24 | +S | `concat(raw, S_24)` | 672 |
| **STL** | 168 | +T | `concat(raw, T_168)` | 672 |
| **STL** | 168 | +S | `concat(raw, S_168)` | 672 |
| **STL-Multi** | [24,168] | +T | `concat(raw, T_24, T_168)` | 1008 |
| **STL-Multi** | [24,168] | +S | `concat(raw, S_24, S_168)` | 1008 |
| **SSA** | w=112 | +T | `concat(raw, T_ssa)` | 672 |
| **SSA** | w=112 | +S | `concat(raw, S_ssa)` | 672 |
| **SSA** | w=168 | +T | `concat(raw, T_ssa)` | 672 |
| **EMD** | - | +T | `concat(raw, IMF_low)` | 672 |
| **EMD** | - | +S | `concat(raw, IMF_mid)` | 672 |
| **Wavelet** | db4,L=3 | +T | `concat(raw, approx)` | 672 |
| **Wavelet** | db4,L=3 | +S | `concat(raw, details)` | 672 |
| **STD_MULTI** | bs=24 | +T | `concat(raw, T_std)` | 672 |
| **STD_MULTI** | bs=24 | +S | `concat(raw, S_std)` | 672 |

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

### Phase 3: 单尺度分解实验
#### 3.1 STL 分解（单尺度）
- [ ] STL-24: 所有数据集 × (+T, +S) × 4 horizons
- [ ] STL-168: 所有数据集 × (+T, +S) × 4 horizons

#### 3.2 SSA 分解（不同窗口）
- [ ] SSA-w112: 所有数据集 × (+T, +S) × 4 horizons
- [ ] SSA-w168: 所有数据集 × (+T, +S) × 4 horizons
- [ ] 验证 SSA 成分分组策略（trend_indices, seasonal_indices）

#### 3.3 其他分解方法
- [ ] EMD: 所有数据集 × (+T, +S) × 4 horizons
- [ ] Wavelet: 所有数据集 × (+T, +S) × 4 horizons
- [ ] STD_MULTI: 所有数据集 × (+T, +S) × 4 horizons
- [ ] MA_BASELINE: 所有数据集 × (+T, +S) × 4 horizons

### Phase 4: 多尺度分解实验（可选）
- [ ] **STL-Multi-T**: `concat(raw, T_24, T_168)` - 多尺度趋势
- [ ] **STL-Multi-S**: `concat(raw, S_24, S_168)` - 多尺度季节
- [ ] **STL-Multi-TS**: `concat(raw, T_24, S_24)` - 单尺度混合
- [ ] **STL-Cross**: `concat(raw, T_24, S_168)` - 跨尺度组合

### Phase 5: 结果分析
- [ ] 汇总所有实验结果到统一 CSV
- [ ] 计算各成分相对 Baseline 的提升/下降
- [ ] 对比单尺度 vs 多尺度性能
- [ ] 分析 SSA 窗口大小的影响
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
- **假设5**: 多尺度趋势 (Multi-T) 比单尺度趋势性能更好
- **假设6**: SSA 窗口大小影响成分质量：window=336/3 优于 336/2
- **假设7**: Per-Window 分解（当前）对自适应方法（EMD/Wavelet）更有效
- **假设8**: 多尺度组合 (Multi-TS) 在长期预测中表现最佳

### 关键研究问题
1. **分解模式选择**：Per-Window vs Global Pattern 哪种更适合实际预测？
2. **多尺度增益**：多尺度成分是否比单尺度显著提升性能？
3. **SSA 参数敏感性**：窗口大小和成分分组策略如何影响结果？
4. **成分互补性**：趋势+季节性组合是否优于单独使用？
5. **方法适用性**：哪些方法在哪些数据集/horizon上表现最佳？

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
