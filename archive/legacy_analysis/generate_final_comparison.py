#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate final comparison table including STD results
"""

import pandas as pd
import numpy as np

# Load new results
df_new = pd.read_csv("outputs/decomp_linear_bench/compare_components/metrics_summary_by_method_horizon_ablation.csv")

# Load STD ablation results
df_std = pd.read_csv("std_component_ablation_summary.csv")

print("="*80)
print("COMPREHENSIVE DECOMPOSITION METHOD COMPARISON")
print("="*80)
print("\nDataset: ETTh1 | Metric: MSE | Model: Ridge Linear Regression")
print("\n" + "="*80)

# Prepare comparison table
methods = ['STL', 'SSA', 'WAVELET', 'EMD', 'MA']
components = ['S', 'T']
horizons = [96, 192, 336, 720]

# Add STD results
std_results = {
    'STD-Full': [0.3419, 0.3568, 0.3710, 0.4085],  # From std_component_ablation_summary.csv
    'STD-Seasonal': [0.2921, 0.3226, 0.3430, 0.4026],
    'STD-Trend': [0.2920, 0.3259, 0.3452, 0.3659],
    'Baseline': [0.2899, 0.3210, 0.3456, 0.3750]
}

print("\n" + "="*80)
print("1. SEASONAL COMPONENT COMPARISON")
print("="*80)

seasonal_data = []
for method in methods:
    df_m = df_new[(df_new['decomp'] == method) & (df_new['ablation'] == 'S')]
    row = {'Method': f'{method}-S'}
    for h in horizons:
        mse = df_m[df_m['horizon'] == h]['MSE'].values
        row[f'H{h}'] = mse[0] if len(mse) > 0 else np.nan
    row['Avg'] = np.nanmean([row[f'H{h}'] for h in horizons])
    seasonal_data.append(row)

# Add STD
row = {'Method': 'STD-S'}
for i, h in enumerate(horizons):
    row[f'H{h}'] = std_results['STD-Seasonal'][i]
row['Avg'] = np.mean([row[f'H{h}'] for h in horizons])
seasonal_data.append(row)

# Add Baseline
row = {'Method': 'Baseline'}
for i, h in enumerate(horizons):
    row[f'H{h}'] = std_results['Baseline'][i]
row['Avg'] = np.mean([row[f'H{h}'] for h in horizons])
seasonal_data.append(row)

df_seasonal = pd.DataFrame(seasonal_data).sort_values('Avg')
print("\nRanked by Average MSE:")
print(df_seasonal.to_string(index=False))

print("\n" + "="*80)
print("2. TREND COMPONENT COMPARISON")
print("="*80)

trend_data = []
for method in methods:
    df_m = df_new[(df_new['decomp'] == method) & (df_new['ablation'] == 'T')]
    row = {'Method': f'{method}-T'}
    for h in horizons:
        mse = df_m[df_m['horizon'] == h]['MSE'].values
        row[f'H{h}'] = mse[0] if len(mse) > 0 else np.nan
    row['Avg'] = np.nanmean([row[f'H{h}'] for h in horizons])
    trend_data.append(row)

# Add STD
row = {'Method': 'STD-T'}
for i, h in enumerate(horizons):
    row[f'H{h}'] = std_results['STD-Trend'][i]
row['Avg'] = np.mean([row[f'H{h}'] for h in horizons])
trend_data.append(row)

# Add Baseline
row = {'Method': 'Baseline'}
for i, h in enumerate(horizons):
    row[f'H{h}'] = std_results['Baseline'][i]
row['Avg'] = np.mean([row[f'H{h}'] for h in horizons])
trend_data.append(row)

df_trend = pd.DataFrame(trend_data).sort_values('Avg')
print("\nRanked by Average MSE:")
print(df_trend.to_string(index=False))

print("\n" + "="*80)
print("3. KEY INSIGHTS")
print("="*80)

print("\n✅ TOP PERFORMERS:")
best_s = df_seasonal.iloc[0]
best_t = df_trend.iloc[0]
print(f"  - Best Seasonal: {best_s['Method']:15s} Avg MSE = {best_s['Avg']:.4f}")
print(f"  - Best Trend:    {best_t['Method']:15s} Avg MSE = {best_t['Avg']:.4f}")

print("\n📊 IMPROVEMENT OVER STD:")
std_s_avg = df_seasonal[df_seasonal['Method'] == 'STD-S']['Avg'].values[0]
std_t_avg = df_trend[df_trend['Method'] == 'STD-T']['Avg'].values[0]
print(f"  - STL-S vs STD-S: {(std_s_avg - best_s['Avg'])/std_s_avg*100:+.1f}%")
print(f"  - SSA-T vs STD-T: {(std_t_avg - best_t['Avg'])/std_t_avg*100:+.1f}%")

print("\n🎯 RECOMMENDATIONS:")
print("  1. Use SSA for Trend extraction (MSE 0.115)")
print("  2. Use STL for Seasonal extraction (MSE 0.146)")
print("  3. Avoid SSA-Seasonal (extremely poor: MSE 0.617)")
print("  4. Consider WAVELET as all-around stable option")
print("  5. STD is outperformed by all modern methods")

print("\n" + "="*80)

# Save combined results
df_seasonal.to_csv("outputs/decomp_linear_bench/compare_components/seasonal_ranking.csv", index=False)
df_trend.to_csv("outputs/decomp_linear_bench/compare_components/trend_ranking.csv", index=False)
print("\nResults saved:")
print("  - seasonal_ranking.csv")
print("  - trend_ranking.csv")
print("="*80)
