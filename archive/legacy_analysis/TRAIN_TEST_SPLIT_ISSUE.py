"""
Critical Issue: Train/Test Split Misalignment

PROBLEM IDENTIFIED:
===================
Current implementation has MULTIPLE fairness issues:

1. ❌ RAW features missing in single-scale methods
2. ❌ Train/test splits NOT aligned across experiments
3. ❌ Different horizons create different sample counts

CURRENT BUGGY FLOW:
===================
For each (method, horizon) pair:
  1. build_dataset(series, lookback, horizon, cfg)
     → Creates n_samples = len(series) - lookback - horizon + 1
     → Different horizon → different n_samples!
  
  2. Split at 80%:
     n_train = int(len(X_full) * 0.8)
     → Different n_samples → different split points!
  
  3. Train and evaluate
     → Each experiment uses DIFFERENT time windows!

EXAMPLE OF THE BUG:
===================
ETTh1 dataset: 17,420 time steps
lookback = 336

Method: STD_MULTI, horizon=96:
  n_samples = 17420 - 336 - 96 + 1 = 16989
  train_size = int(16989 * 0.8) = 13591
  train_indices: [0:13591]
  test_indices: [13591:16989]
  
Method: STD_MULTI, horizon=720:
  n_samples = 17420 - 336 - 720 + 1 = 16365
  train_size = int(16365 * 0.8) = 13092
  train_indices: [0:13092]  ← DIFFERENT!
  test_indices: [13092:16365]  ← DIFFERENT!

Result: Comparing apples to oranges!

CORRECT APPROACH:
=================
Split the TIME SERIES first, then create samples:

```python
# Step 1: Split raw series at TIME POINT (not sample count)
total_len = len(series)
train_end_idx = int(total_len * 0.8)  # e.g., 13936 for ETTh1

train_series = series[:train_end_idx]
test_series = series[train_end_idx:]

# Step 2: Create samples from respective splits
def create_samples(series_segment, lookback, horizon):
    n_samples = len(series_segment) - lookback - horizon + 1
    samples = []
    for i in range(n_samples):
        X = series_segment[i:i+lookback]
        y = series_segment[i+lookback+horizon-1]
        samples.append((X, y))
    return samples

train_samples = create_samples(train_series, lookback, horizon)
test_samples = create_samples(test_series, lookback, horizon)
```

KEY DIFFERENCES:
================
WRONG (current):
  - Split = 80% of SAMPLES (varies by horizon)
  - train_end = sample_index(varying)
  
CORRECT (should be):
  - Split = 80% of TIME SERIES (fixed time point)
  - train_end = time_index(13936 for ETTh1)

VERIFICATION:
=============
After fix, all experiments should have:
  Same train_end time index: 13936
  Same test_start time index: 13936
  Different sample counts per horizon (expected)
  
horizon=96:  train_samples ≈ 13504, test_samples ≈ 3485
horizon=720: train_samples ≈ 12880, test_samples ≈ 2861

But they all use the SAME time range for training!

REQUIRED CHANGES:
=================
1. Modify runner.py:
   - Split series ONCE at start of run_experiment()
   - Pass train_series and test_series to run_single_experiment()
   
2. Modify builder.py:
   - Accept series_segment instead of full series
   - Remove internal splitting logic
   
3. Update all experiments:
   - Re-run ALL experiments with fixed splitting
   - Old results are INVALID for comparison

IMPACT:
=======
Current results are UNRELIABLE because:
- Different experiments trained on different time periods
- Horizon=96 uses MORE training data than horizon=720
- Makes horizon comparison meaningless
- Makes method comparison questionable

NEXT STEPS:
===========
1. Fix train/test splitting logic
2. Fix RAW feature inclusion
3. Re-run ALL experiments from scratch
4. Then do fair comparison
"""

if __name__ == "__main__":
    print(__doc__)
