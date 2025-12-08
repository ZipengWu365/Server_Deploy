"""
Analyze ETTh1 and ETTh2 results to compare Seasonal vs Trend components across methods.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Read results
df1 = pd.read_csv('outputs/decomp_linear_bench/etth1_all_methods/metrics_summary_by_method_horizon_ablation.csv')
df2 = pd.read_csv('outputs/decomp_linear_bench/etth2_all_methods/metrics_summary_by_method_horizon_ablation.csv')

# Combine datasets
df = pd.concat([df1, df2], ignore_index=True)

# Remove failed methods (n_feat == 0)
df = df[df['n_feat'] > 0].copy()

print("=" * 80)
print("Cross-Dataset Analysis: ETTh1 and ETTh2")
print("=" * 80)
print(f"\nTotal experiments: {len(df)}")
print(f"Datasets: {df['dataset'].unique()}")
print(f"Methods: {sorted(df['decomp'].unique())}")
print(f"Ablations: {sorted(df['ablation'].unique())}")
print(f"Horizons: {sorted(df['horizon'].unique())}")

# ============================================================================
# 1. Component Performance Ranking (Overall)
# ============================================================================
print("\n" + "=" * 80)
print("1. OVERALL COMPONENT PERFORMANCE RANKING (Lower MSE = Better)")
print("=" * 80)

component_stats = df.groupby(['decomp', 'ablation']).agg({
    'MSE': ['mean', 'std', 'min', 'max'],
    'MAE': ['mean', 'std'],
    'R2': ['mean']
}).round(4)

# Flatten column names
component_stats.columns = ['_'.join(col).strip() for col in component_stats.columns.values]
component_stats = component_stats.reset_index()
component_stats = component_stats.sort_values('MSE_mean')

print("\nRanking by MSE (mean across all horizons and datasets):")
print(component_stats[['decomp', 'ablation', 'MSE_mean', 'MSE_std', 'MAE_mean', 'R2_mean']].to_string(index=False))

# ============================================================================
# 2. Best Component by Method
# ============================================================================
print("\n" + "=" * 80)
print("2. SEASONAL vs TREND: Which Component Performs Better per Method?")
print("=" * 80)

method_component_comp = []
for method in df['decomp'].unique():
    method_df = df[df['decomp'] == method]
    
    seasonal_mse = method_df[method_df['ablation'] == 'S']['MSE'].mean()
    trend_mse = method_df[method_df['ablation'] == 'T']['MSE'].mean()
    
    seasonal_mae = method_df[method_df['ablation'] == 'S']['MAE'].mean()
    trend_mae = method_df[method_df['ablation'] == 'T']['MAE'].mean()
    
    best_component = 'Trend' if trend_mse < seasonal_mse else 'Seasonal'
    mse_gap = abs(seasonal_mse - trend_mse)
    
    method_component_comp.append({
        'Method': method,
        'Seasonal_MSE': round(seasonal_mse, 4),
        'Trend_MSE': round(trend_mse, 4),
        'Best_Component': best_component,
        'MSE_Gap': round(mse_gap, 4),
        'Gap_%': round(100 * mse_gap / max(seasonal_mse, trend_mse), 2)
    })

comp_df = pd.DataFrame(method_component_comp).sort_values('Trend_MSE')
print("\n" + comp_df.to_string(index=False))

print("\n📊 Key Insight:")
trend_better_count = sum(1 for x in method_component_comp if x['Best_Component'] == 'Trend')
print(f"   - Trend components are better in {trend_better_count}/{len(method_component_comp)} methods")
print(f"   - Average MSE gap: {comp_df['MSE_Gap'].mean():.4f}")

# ============================================================================
# 3. Best Method for Each Component Type
# ============================================================================
print("\n" + "=" * 80)
print("3. BEST METHOD FOR EACH COMPONENT TYPE")
print("=" * 80)

seasonal_best = df[df['ablation'] == 'S'].groupby('decomp')['MSE'].mean().sort_values()
trend_best = df[df['ablation'] == 'T'].groupby('decomp')['MSE'].mean().sort_values()

print("\n📈 Seasonal Component Ranking (MSE):")
for i, (method, mse) in enumerate(seasonal_best.items(), 1):
    print(f"   {i}. {method}: {mse:.4f}")

print("\n📉 Trend Component Ranking (MSE):")
for i, (method, mse) in enumerate(trend_best.items(), 1):
    print(f"   {i}. {method}: {mse:.4f}")

# ============================================================================
# 4. Dataset-Specific Patterns
# ============================================================================
print("\n" + "=" * 80)
print("4. DATASET-SPECIFIC PATTERNS")
print("=" * 80)

for dataset in ['ETTh1', 'ETTh2']:
    print(f"\n{dataset}:")
    ds_df = df[df['dataset'] == dataset]
    
    # Best overall
    best_row = ds_df.loc[ds_df['MSE'].idxmin()]
    print(f"   Best: {best_row['decomp']}-{best_row['ablation']} @ h={best_row['horizon']} (MSE: {best_row['MSE']:.4f})")
    
    # Best seasonal
    seasonal_df = ds_df[ds_df['ablation'] == 'S']
    best_s = seasonal_df.loc[seasonal_df['MSE'].idxmin()]
    print(f"   Best Seasonal: {best_s['decomp']} @ h={best_s['horizon']} (MSE: {best_s['MSE']:.4f})")
    
    # Best trend
    trend_df = ds_df[ds_df['ablation'] == 'T']
    best_t = trend_df.loc[trend_df['MSE'].idxmin()]
    print(f"   Best Trend: {best_t['decomp']} @ h={best_t['horizon']} (MSE: {best_t['MSE']:.4f})")

# ============================================================================
# 5. Horizon Stability Analysis
# ============================================================================
print("\n" + "=" * 80)
print("5. HORIZON STABILITY (MSE Std across horizons)")
print("=" * 80)

stability = df.groupby(['decomp', 'ablation'])['MSE'].std().reset_index()
stability.columns = ['Method', 'Component', 'MSE_Std']
stability = stability.sort_values('MSE_Std')

print("\nMost Stable (Lower std = More consistent across horizons):")
print(stability.to_string(index=False))

# ============================================================================
# 6. Final Recommendations
# ============================================================================
print("\n" + "=" * 80)
print("6. 🎯 FINAL RECOMMENDATIONS")
print("=" * 80)

print("\n✅ For Seasonal Component:")
best_seasonal_method = seasonal_best.index[0]
best_seasonal_mse = seasonal_best.iloc[0]
print(f"   → Use: {best_seasonal_method} (MSE: {best_seasonal_mse:.4f})")

print("\n✅ For Trend Component:")
best_trend_method = trend_best.index[0]
best_trend_mse = trend_best.iloc[0]
print(f"   → Use: {best_trend_method} (MSE: {best_trend_mse:.4f})")

print("\n✅ Overall Best Component to Use:")
if best_trend_mse < best_seasonal_mse:
    gap = best_seasonal_mse - best_trend_mse
    improvement = 100 * gap / best_seasonal_mse
    print(f"   → Trend components are {improvement:.1f}% better than Seasonal")
    print(f"   → Best choice: {best_trend_method}-T (MSE: {best_trend_mse:.4f})")
else:
    gap = best_trend_mse - best_seasonal_mse
    improvement = 100 * gap / best_trend_mse
    print(f"   → Seasonal components are {improvement:.1f}% better than Trend")
    print(f"   → Best choice: {best_seasonal_method}-S (MSE: {best_seasonal_mse:.4f})")

print("\n✅ Method Versatility (Good for both S and T):")
versatility = []
for method in df['decomp'].unique():
    s_mse = seasonal_best.get(method, np.inf)
    t_mse = trend_best.get(method, np.inf)
    avg_mse = (s_mse + t_mse) / 2
    gap = abs(s_mse - t_mse)
    versatility.append({'Method': method, 'Avg_MSE': avg_mse, 'S_T_Gap': gap, 'Balance_Score': gap / avg_mse})

vers_df = pd.DataFrame(versatility).sort_values('Balance_Score')
print(f"   → Most balanced: {vers_df.iloc[0]['Method']} (S-T gap: {vers_df.iloc[0]['S_T_Gap']:.4f})")

# ============================================================================
# 7. Visualization
# ============================================================================
print("\n" + "=" * 80)
print("7. Generating Visualizations...")
print("=" * 80)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Component comparison heatmap
pivot_data = df.groupby(['decomp', 'ablation'])['MSE'].mean().unstack()
sns.heatmap(pivot_data, annot=True, fmt='.4f', cmap='RdYlGn_r', ax=axes[0, 0], cbar_kws={'label': 'MSE'})
axes[0, 0].set_title('Average MSE: Seasonal vs Trend per Method', fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel('Component (S=Seasonal, T=Trend)')
axes[0, 0].set_ylabel('Decomposition Method')

# Plot 2: Method ranking by component
methods = sorted(df['decomp'].unique())
s_mses = [seasonal_best.get(m, np.nan) for m in methods]
t_mses = [trend_best.get(m, np.nan) for m in methods]

x = np.arange(len(methods))
width = 0.35
axes[0, 1].bar(x - width/2, s_mses, width, label='Seasonal', alpha=0.8)
axes[0, 1].bar(x + width/2, t_mses, width, label='Trend', alpha=0.8)
axes[0, 1].set_xlabel('Method')
axes[0, 1].set_ylabel('Average MSE')
axes[0, 1].set_title('Seasonal vs Trend Performance by Method', fontsize=14, fontweight='bold')
axes[0, 1].set_xticks(x)
axes[0, 1].set_xticklabels(methods, rotation=45)
axes[0, 1].legend()
axes[0, 1].grid(axis='y', alpha=0.3)

# Plot 3: Horizon stability
for method in methods:
    for comp in ['S', 'T']:
        method_comp_df = df[(df['decomp'] == method) & (df['ablation'] == comp)]
        if len(method_comp_df) > 0:
            horizons_mse = method_comp_df.groupby('horizon')['MSE'].mean()
            axes[1, 0].plot(horizons_mse.index, horizons_mse.values, marker='o', label=f'{method}-{comp}', alpha=0.7)

axes[1, 0].set_xlabel('Forecast Horizon')
axes[1, 0].set_ylabel('MSE')
axes[1, 0].set_title('MSE across Forecast Horizons', fontsize=14, fontweight='bold')
axes[1, 0].legend(loc='upper left', fontsize=8)
axes[1, 0].grid(alpha=0.3)

# Plot 4: Dataset comparison
for dataset in ['ETTh1', 'ETTh2']:
    ds_df = df[df['dataset'] == dataset]
    comp_mse = ds_df.groupby('ablation')['MSE'].mean()
    axes[1, 1].bar([f'{dataset}-S', f'{dataset}-T'], [comp_mse.get('S', 0), comp_mse.get('T', 0)], alpha=0.7, label=dataset)

axes[1, 1].set_ylabel('Average MSE')
axes[1, 1].set_title('Component Performance by Dataset', fontsize=14, fontweight='bold')
axes[1, 1].legend()
axes[1, 1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/decomp_linear_bench/etth_component_analysis.png', dpi=300, bbox_inches='tight')
print("✅ Saved: outputs/decomp_linear_bench/etth_component_analysis.png")

print("\n" + "=" * 80)
print("Analysis Complete!")
print("=" * 80)
