"""
Verify fair comparison: analyze exact input features for each experiment
"""
import pandas as pd
import numpy as np
from pathlib import Path

def analyze_experiment_features():
    """
    Analyze and compare feature configurations across all experiments
    """
    
    print("=" * 80)
    print("FEATURE CONFIGURATION COMPARISON - Fair vs Unfair?")
    print("=" * 80)
    
    # Load unified results
    results_path = Path("outputs/decomp_linear_bench/unified_analysis/unified_all_results.csv")
    df = pd.read_csv(results_path)
    
    # Group by method to see feature counts
    feature_summary = df.groupby(['Method', 'Category']).agg({
        'Features': ['first', 'std'],
        'MSE': 'mean'
    }).round(4)
    
    print("\n" + "="*80)
    print("FEATURE COUNT BY METHOD")
    print("="*80)
    print(feature_summary)
    
    # Detailed breakdown by experiment type
    print("\n" + "="*80)
    print("DETAILED FEATURE COMPOSITION ANALYSIS")
    print("="*80)
    
    # STD experiments
    print("\n📊 STD EXPERIMENTS (Multi-Scale & Full Decomposition):")
    print("-" * 80)
    
    std_configs = [
        {
            'name': 'STD_MULTI-Trend',
            'features': 1348,
            'composition': 'lookback_raw(336) + multi_scale_trend(336) + trend_slope(1) + 3×scales×stats',
            'ablation': 'T (Trend only)',
            'formula': 'Raw lookback + Trend components + Trend metadata'
        },
        {
            'name': 'STD_MULTI-Season', 
            'features': 1348,
            'composition': 'lookback_raw(336) + multi_scale_season(336) + season_energy(1) + 3×scales×stats',
            'ablation': 'S (Season only)',
            'formula': 'Raw lookback + Seasonal components + Seasonal metadata'
        },
        {
            'name': 'STD_FULL-Trend',
            'features': 1011,
            'composition': 'lookback_raw(336) + full_trend(336) + trend_slope(1) + scales×stats',
            'ablation': 'T (Trend only)',
            'formula': 'Raw lookback + Full-scale trend + Trend metadata'
        },
        {
            'name': 'STD_FULL-Season',
            'features': 1011,
            'composition': 'lookback_raw(336) + full_season(336) + season_energy(1) + scales×stats',
            'ablation': 'S (Season only)',
            'formula': 'Raw lookback + Full-scale seasonal + Seasonal metadata'
        },
        {
            'name': 'STD_MULTI-Full',
            'features': 4044,
            'composition': 'lookback_raw(336) + multi_trend(336) + multi_season(336) + residual(336) + all_stats',
            'ablation': 'T+S+D (All components)',
            'formula': 'Raw + All multi-scale components'
        },
        {
            'name': 'STD_FULL-Full',
            'features': 3033,
            'composition': 'lookback_raw(336) + full_trend(336) + full_season(336) + residual(336) + stats',
            'ablation': 'T+S+D (All components)',
            'formula': 'Raw + All full-scale components'
        }
    ]
    
    for cfg in std_configs:
        print(f"\n  {cfg['name']}:")
        print(f"    Features: {cfg['features']}")
        print(f"    Ablation: {cfg['ablation']}")
        print(f"    Formula: {cfg['formula']}")
        print(f"    Details: {cfg['composition']}")
    
    # Single-scale experiments  
    print("\n" + "="*80)
    print("📊 SINGLE-SCALE EXPERIMENTS (STL, SSA, Wavelet, EMD):")
    print("-" * 80)
    
    single_configs = [
        {
            'name': 'STL-T / SSA-T / WAVELET-T / EMD-T',
            'features': 337,
            'composition': 'trend(336) + trend_slope(1)',
            'ablation': 'T (Trend ONLY - NO RAW)',
            'formula': 'ONLY Trend component + metadata',
            'note': '⚠️ MISSING RAW LOOKBACK!'
        },
        {
            'name': 'STL-S / SSA-S / WAVELET-S / EMD-S',
            'features': 337,
            'composition': 'season(336) + season_energy(1)',
            'ablation': 'S (Season ONLY - NO RAW)',
            'formula': 'ONLY Seasonal component + metadata',
            'note': '⚠️ MISSING RAW LOOKBACK!'
        }
    ]
    
    for cfg in single_configs:
        print(f"\n  {cfg['name']}:")
        print(f"    Features: {cfg['features']}")
        print(f"    Ablation: {cfg['ablation']}")
        print(f"    Formula: {cfg['formula']}")
        print(f"    Details: {cfg['composition']}")
        print(f"    {cfg['note']}")
    
    # Key difference analysis
    print("\n" + "="*80)
    print("🚨 CRITICAL FINDING: UNFAIR COMPARISON!")
    print("="*80)
    
    print("""
STD experiments:
  Input = RAW(336) + Component(336) + Stats(~3-340)
  Total: 1011-4044 features
  
Single-scale experiments:
  Input = Component(336) + Stats(1)
  Total: 337 features
  ⚠️ MISSING THE RAW LOOKBACK!!!

This is NOT a fair comparison because:
1. STD has BOTH raw AND decomposed components
2. Single-scale has ONLY decomposed components
3. Raw lookback alone would give baseline performance
4. The comparison conflates "multi-scale" with "raw+component"

CORRECT comparison should be:
- STD_MULTI-T (raw+multi-trend) vs STL-T+RAW (raw+single-trend)
- STD_MULTI-S (raw+multi-season) vs STL-S+RAW (raw+single-season)

Currently comparing:
- STD: raw+component (1348 feat) 
- Single-scale: component only (337 feat)
""")
    
    # Calculate what the fair comparison should be
    print("\n" + "="*80)
    print("📐 EXPECTED FAIR FEATURE COUNTS:")
    print("="*80)
    
    lookback = 336
    
    fair_comparison = pd.DataFrame([
        {'Experiment': 'STD_MULTI-Trend', 'Current': 1348, 'Raw': lookback, 'Component': 336, 'Stats': '~676', 'Fair': '✓'},
        {'Experiment': 'STD_FULL-Trend', 'Current': 1011, 'Raw': lookback, 'Component': 336, 'Stats': '~339', 'Fair': '✓'},
        {'Experiment': 'STL-T (current)', 'Current': 337, 'Raw': 0, 'Component': 336, 'Stats': 1, 'Fair': '✗ Missing RAW'},
        {'Experiment': 'STL-T (should be)', 'Current': '673*', 'Raw': lookback, 'Component': 336, 'Stats': 1, 'Fair': '✓'},
        {'Experiment': 'SSA-T (current)', 'Current': 337, 'Raw': 0, 'Component': 336, 'Stats': 1, 'Fair': '✗ Missing RAW'},
        {'Experiment': 'SSA-T (should be)', 'Current': '673*', 'Raw': lookback, 'Component': 336, 'Stats': 1, 'Fair': '✓'},
    ])
    
    print(fair_comparison.to_string(index=False))
    print("\n*Estimated if RAW was included")
    
    # Performance impact
    print("\n" + "="*80)
    print("📊 PERFORMANCE IMPACT OF MISSING RAW:")
    print("="*80)
    
    # Extract relevant comparisons
    std_multi_t = df[df['Method'] == 'STD_MULTI-Trend']['MSE'].mean()
    stl_t = df[df['Method'] == 'STL-T']['MSE'].mean()
    ssa_t = df[df['Method'] == 'SSA-T']['MSE'].mean()
    
    print(f"\nSTD_MULTI-Trend (raw+multi-trend): {std_multi_t:.4f} MSE")
    print(f"STL-T (trend only):                 {stl_t:.4f} MSE ({stl_t/std_multi_t:.2f}x worse)")
    print(f"SSA-T (trend only):                 {ssa_t:.4f} MSE ({ssa_t/std_multi_t:.2f}x worse)")
    
    print(f"""
Key Question: How much of the {stl_t/std_multi_t:.2f}x performance gap is due to:
  A) Multi-scale decomposition being better?
  B) Including raw lookback features?

To answer this, we need to run:
  - Baseline: RAW only (336 features)
  - STL-T + RAW (336 + 336 + 1 = 673 features)
  - SSA-T + RAW (336 + 336 + 1 = 673 features)
  
Then compare fairly against STD_MULTI-T (1348 features).
""")
    
    # Recommendation
    print("\n" + "="*80)
    print("💡 RECOMMENDATIONS TO FIX COMPARISON:")
    print("="*80)
    
    print("""
1. ✅ KEEP CURRENT STD EXPERIMENTS (already include raw)
   
2. ❌ DISCARD CURRENT SINGLE-SCALE RESULTS (missing raw)

3. ✅ RE-RUN SINGLE-SCALE WITH RAW:
   Modify builder.py to ALWAYS include raw lookback:
   
   Current:
     ablation="T" → only trend component
   
   Should be:
     ablation="T" → raw + trend component
     ablation="S" → raw + seasonal component
   
4. ✅ ADD BASELINE:
   ablation="RAW" → raw lookback only (no decomposition)
   
5. ✅ THEN COMPARE:
   - Baseline (RAW only, 336 feat)
   - STL-T+RAW (673 feat) vs STD_MULTI-T (1348 feat)
   - SSA-T+RAW (673 feat) vs STD_MULTI-T (1348 feat)
   
This will isolate the benefit of multi-scale decomposition from raw features.
""")

if __name__ == "__main__":
    analyze_experiment_features()
