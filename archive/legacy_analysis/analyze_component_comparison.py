#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Analyze cross-method component comparison results
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def load_results(output_dir="outputs/decomp_linear_bench/compare_components/"):
    """Load experiment results"""
    summary_file = os.path.join(output_dir, "metrics_summary_by_method_horizon_ablation.csv")
    
    if not os.path.exists(summary_file):
        print(f"Results file not found: {summary_file}")
        return None
    
    df = pd.read_csv(summary_file)
    return df

def analyze_component_ranking(df):
    """Analyze which method's components perform best"""
    print("\n" + "="*80)
    print("Component Performance Ranking")
    print("="*80)
    
    # Group by component type (S or T) and method
    for component in ['S', 'T']:
        df_comp = df[df['ablation'] == component]
        
        print(f"\n{'='*40}")
        print(f"Component: {component} ({'Seasonal' if component == 'S' else 'Trend'})")
        print(f"{'='*40}")
        
        for horizon in sorted(df_comp['horizon'].unique()):
            df_h = df_comp[df_comp['horizon'] == horizon]
            
            # Sort by MSE
            df_sorted = df_h.sort_values('MSE')
            
            print(f"\nHorizon {horizon}:")
            print(f"  Rank | Method        | MSE      | MAE      | n_feat")
            print(f"  " + "-"*55)
            
            for rank, (idx, row) in enumerate(df_sorted.iterrows(), 1):
                print(f"  {rank:4d} | {row['decomp']:13s} | {row['MSE']:.6f} | {row['MAE']:.6f} | {int(row['n_feat']):6d}")
    
    return df

def compare_seasonal_vs_trend(df):
    """Compare S vs T performance for each method"""
    print("\n" + "="*80)
    print("Seasonal vs Trend Comparison (by Method)")
    print("="*80)
    
    methods = df['decomp'].unique()
    
    for method in methods:
        df_method = df[df['decomp'] == method]
        
        print(f"\n{method}:")
        print(f"  Horizon | S (MSE)   | T (MSE)   | Winner")
        print(f"  " + "-"*45)
        
        for horizon in sorted(df_method['horizon'].unique()):
            df_h = df_method[df_method['horizon'] == horizon]
            
            s_mse = df_h[df_h['ablation'] == 'S']['MSE'].values
            t_mse = df_h[df_h['ablation'] == 'T']['MSE'].values
            
            if len(s_mse) > 0 and len(t_mse) > 0:
                s_mse, t_mse = s_mse[0], t_mse[0]
                winner = "S" if s_mse < t_mse else "T"
                print(f"  {horizon:7d} | {s_mse:.6f} | {t_mse:.6f} | {winner}")

def generate_heatmaps(df, output_dir="outputs/decomp_linear_bench/compare_components/"):
    """Generate heatmap visualizations"""
    print("\n" + "="*80)
    print("Generating Visualizations")
    print("="*80)
    
    # Heatmap 1: Seasonal components across methods and horizons
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    for idx, component in enumerate(['S', 'T']):
        df_comp = df[df['ablation'] == component]
        pivot = df_comp.pivot_table(values='MSE', index='decomp', columns='horizon')
        
        sns.heatmap(pivot, annot=True, fmt='.4f', cmap='RdYlGn_r', ax=axes[idx],
                   cbar_kws={'label': 'MSE'})
        axes[idx].set_title(f"{'Seasonal' if component == 'S' else 'Trend'} Component Performance",
                          fontsize=14, fontweight='bold')
        axes[idx].set_xlabel('Horizon', fontsize=12)
        axes[idx].set_ylabel('Method', fontsize=12)
    
    plt.tight_layout()
    out_file = os.path.join(output_dir, "component_comparison_heatmap.png")
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    print(f"  Saved: {out_file}")
    plt.close()
    
    # Heatmap 2: S vs T comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    
    methods = df['decomp'].unique()
    horizons = sorted(df['horizon'].unique())
    
    s_vs_t_matrix = []
    for method in methods:
        row = []
        for horizon in horizons:
            df_mh = df[(df['decomp'] == method) & (df['horizon'] == horizon)]
            s_mse = df_mh[df_mh['ablation'] == 'S']['MSE'].values
            t_mse = df_mh[df_mh['ablation'] == 'T']['MSE'].values
            
            if len(s_mse) > 0 and len(t_mse) > 0:
                # Positive: T better, Negative: S better
                diff = s_mse[0] - t_mse[0]
                row.append(diff)
            else:
                row.append(0)
        s_vs_t_matrix.append(row)
    
    s_vs_t_df = pd.DataFrame(s_vs_t_matrix, index=methods, columns=horizons)
    
    sns.heatmap(s_vs_t_df, annot=True, fmt='.4f', cmap='RdBu', center=0, ax=ax,
               cbar_kws={'label': 'MSE Difference (S - T)'})
    ax.set_title('Seasonal vs Trend Performance\n(Negative = S Better, Positive = T Better)',
                fontsize=14, fontweight='bold')
    ax.set_xlabel('Horizon', fontsize=12)
    ax.set_ylabel('Method', fontsize=12)
    
    plt.tight_layout()
    out_file = os.path.join(output_dir, "seasonal_vs_trend_diff.png")
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    print(f"  Saved: {out_file}")
    plt.close()

def generate_summary_report(df, output_dir="outputs/decomp_linear_bench/compare_components/"):
    """Generate summary report"""
    report_lines = []
    
    report_lines.append("="*80)
    report_lines.append("CROSS-METHOD COMPONENT COMPARISON - SUMMARY REPORT")
    report_lines.append("="*80)
    report_lines.append(f"Dataset: ETTh1")
    report_lines.append(f"Methods: {', '.join(df['decomp'].unique())}")
    report_lines.append(f"Components: Seasonal (S), Trend (T)")
    report_lines.append(f"Horizons: {sorted(df['horizon'].unique())}")
    report_lines.append("="*80)
    
    # Overall winner
    report_lines.append("\n" + "="*80)
    report_lines.append("OVERALL WINNERS")
    report_lines.append("="*80)
    
    for component in ['S', 'T']:
        df_comp = df[df['ablation'] == component]
        avg_mse = df_comp.groupby('decomp')['MSE'].mean().sort_values()
        
        winner = avg_mse.index[0]
        winner_mse = avg_mse.values[0]
        
        comp_name = "Seasonal" if component == 'S' else "Trend"
        report_lines.append(f"\nBest {comp_name} Component: {winner}")
        report_lines.append(f"  Average MSE: {winner_mse:.6f}")
        report_lines.append(f"  Ranking:")
        for rank, (method, mse) in enumerate(avg_mse.items(), 1):
            report_lines.append(f"    {rank}. {method:10s}: {mse:.6f}")
    
    report_text = "\n".join(report_lines)
    
    out_file = os.path.join(output_dir, "component_comparison_summary.txt")
    with open(out_file, 'w') as f:
        f.write(report_text)
    
    print(report_text)
    print(f"\n  Report saved: {out_file}")

def main():
    df = load_results()
    
    if df is None:
        print("No results to analyze. Please run the experiment first.")
        return
    
    print(f"Loaded {len(df)} experiment results")
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nPreview:")
    print(df.head())
    
    # Analysis
    analyze_component_ranking(df)
    compare_seasonal_vs_trend(df)
    generate_heatmaps(df)
    generate_summary_report(df)
    
    print("\n" + "="*80)
    print("Analysis Complete!")
    print("="*80)

if __name__ == "__main__":
    main()
