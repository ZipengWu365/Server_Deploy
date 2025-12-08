# 分解模式对比：Per-Window vs Global Pattern

## 🔴 核心问题

**在测试集上预测时，是否需要重新对每个lookback窗口进行分解？**

这个问题直接影响实验设计和实际应用场景的匹配度。

---

## 两种分解模式详解

### 模式A：Per-Window Decomposition（当前实现）

#### 工作流程
```python
# 训练阶段
for train_window in train_sliding_windows:
    trend, seasonal = decompose(train_window)  # 每个窗口独立分解
    feature = concat(train_window, trend, seasonal)
    model.fit(feature, target)

# 测试阶段
for test_window in test_sliding_windows:
    trend, seasonal = decompose(test_window)  # 测试窗口也重新分解
    feature = concat(test_window, trend, seasonal)
    prediction = model.predict(feature)
```

#### 特点
- ✅ **优点**：
  - 每个窗口获得最适应当前数据的分解
  - 能捕捉局部时间模式的变化
  - 适用于非平稳序列
  
- ❌ **缺点**：
  - **计算量大**：每个测试窗口都需要分解
  - **数据泄露风险**：如果分解算法"看到"了整个窗口的未来信息
  - **不符合实时预测**：实际预测时可能无法等待完整窗口

#### 适用方法
- **EMD**：自适应分解，每个窗口的模态不同
- **Wavelet**：小波系数取决于窗口内容
- **STD_MULTI**：自适应季节趋势分解

---

### 模式B：Global Pattern Decomposition（推荐）

#### 工作流程
```python
# 训练阶段：只分解一次
full_train_trend, full_train_seasonal = decompose(entire_train_sequence)

# 提取训练窗口特征
for i, train_window in enumerate(train_sliding_windows):
    trend_slice = full_train_trend[i : i+lookback]
    seasonal_slice = full_train_seasonal[i : i+lookback]
    feature = concat(train_window, trend_slice, seasonal_slice)
    model.fit(feature, target)

# 测试阶段：使用训练集分解的延续
for i, test_window in enumerate(test_sliding_windows):
    # 注意：这里需要完整序列（train+test）的分解
    # 或者使用训练集学到的周期模式外推
    trend_slice = full_trend[train_len+i : train_len+i+lookback]
    seasonal_slice = full_seasonal[train_len+i : train_len+i+lookback]
    feature = concat(test_window, trend_slice, seasonal_slice)
    prediction = model.predict(feature)
```

#### 特点
- ✅ **优点**：
  - **计算高效**：只需要分解一次
  - **符合实际**：周期模式在训练时确定，测试时直接使用
  - **避免数据泄露**：测试集分解使用的是全局模式，不依赖测试窗口本身
  
- ❌ **缺点**：
  - **假设模式固定**：要求趋势和周期在测试集上保持一致
  - **需要完整序列**：要么对train+test一起分解（不现实），要么外推周期模式

#### 适用方法
- **STL**：假设周期固定，可以用训练集的周期参数外推
- **SSA**：主成分和周期性在训练集上确定
- **MSTL**：多周期模式固定

---

## 🤔 问题：测试集分解的数据泄露

### 场景1：Per-Window分解（当前）
```python
test_window = [t-336, t-335, ..., t-1]  # 336个历史点
trend, seasonal = decompose(test_window)  # 分解这336个点

# 问题：分解算法是否使用了整个窗口的信息？
# - STL: 是的，使用了整个窗口来估计趋势（LOESS平滑）
# - SSA: 是的，SVD分解使用了整个窗口的协方差矩阵
# - EMD: 是的，需要整个窗口来识别IMF
```

**这不算严格的"数据泄露"**，因为：
- 分解只使用了lookback窗口内的数据（历史数据）
- 没有使用预测目标（未来数据）

但是，**在实际应用中可能不现实**：
- 实时预测时，可能无法等待完整的336个点才开始分解
- 每次预测都重新分解，计算成本太高

### 场景2：Global Pattern分解
```python
# 训练阶段
train_trend, train_seasonal = decompose(train_sequence)

# 测试阶段（两种方案）
# 方案1：对完整序列（train+test）分解
full_trend, full_seasonal = decompose(concat(train, test))  # ❌ 数据泄露！

# 方案2：外推周期模式
seasonal_period = estimate_period_from_train(train_seasonal)
test_seasonal = extrapolate_seasonal(seasonal_period, len(test))  # ✅ 合理
test_trend = extrapolate_trend(train_trend, len(test))  # ✅ 合理
```

方案2才是真正"无数据泄露"的全局模式方法。

---

## 🎯 推荐实验设计

### 第一阶段：Per-Window（当前）
- 保持当前实现，测试所有方法
- 理解为"理想情况"：有足够时间对每个窗口分解
- 适用场景：离线批量预测、计算资源充足

### 第二阶段：Global Pattern（未来）
- 实现"训练集分解 + 模式外推"
- 更符合实时预测场景
- 比较两种模式的性能差异

### 实验对比
| 方法 | Per-Window | Global Pattern | 差异 |
|------|------------|----------------|------|
| STL-24 | 每窗口分解 | 训练集周期外推 | 对比周期稳定性假设的影响 |
| SSA | 每窗口SVD | 训练集主成分投影 | 对比局部vs全局主成分 |
| EMD | 每窗口IMF | ❌ 不适用 | EMD必须用Per-Window |

---

## 📊 预期发现

### 假设1：STL/SSA在两种模式下性能相近
- 如果周期稳定，Global Pattern应该和Per-Window接近
- 如果性能下降明显，说明测试集周期模式发生变化

### 假设2：EMD/Wavelet只能用Per-Window
- 自适应方法无法外推全局模式
- 必须对每个窗口重新分解

### 假设3：实时预测更适合Global Pattern
- 计算效率高
- 但需要假设周期稳定

---

## 🔧 实现建议

### 当前阶段：保持Per-Window
理由：
1. 已经实现，可以快速获得实验结果
2. 避免数据泄露（使用lookback窗口，不用未来数据）
3. 理解为"离线批量预测"场景

### 未来扩展：添加Global Pattern
需要实现：
1. `decompose_global_pattern(train_data, test_len)` - 训练集分解 + 外推
2. 新增配置参数 `decomposition_mode: "per_window" | "global_pattern"`
3. 对比两种模式的性能差异

---

## 💡 关键结论

### 当前实现（Per-Window）
- **不是数据泄露**：只使用lookback窗口的历史数据
- **场景匹配**：适合离线批量预测，有充足时间分解
- **计算成本**：高（每个窗口都分解）

### 推荐改进（Global Pattern）
- **更真实**：符合实时预测场景（周期模式预先确定）
- **更高效**：只需分解一次训练集
- **需要假设**：周期和趋势模式在测试集上保持稳定

### 实验建议
1. **先跑Per-Window实验**（当前实现）- 获得基准结果
2. **分析周期稳定性** - 检查测试集的周期是否变化
3. **再实现Global Pattern** - 对比两种模式的性能
4. **根据应用场景选择** - 实时预测用Global，离线预测用Per-Window
