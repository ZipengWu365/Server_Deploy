#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Cross-dataset analysis: Compare method performance across all datasets
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Result directories
RESULT_DIRS = {
    'ETTh1': 'outputs/decomp_linear_bench/etth1_all_methods/',
    'ETTh2': 'outputs/decomp_linear_bench/etth2_all_methods/',
    'ETTm1': 'outputs/decomp_linear_bench/ettm1_all_methods/',
    'ETTm2': 'outputs/decomp_linear_bench/ettm2_all_methods/',
    'Exchange': 'outputs/decomp_linear_bench/exchange_all_methods/',
}

def load_all_results():
    """Load results from all datasets"""
    all_data = []
    
    for dataset_name, result_dir in RESULT_DIRS.items():
        csv_file = os.path.join(result_dir, "metrics_summary_by_method_horizon_ablation.csv")
        
        if not os.path.exists(csv_file):
            print(f"⚠️ Results not found for {dataset_name}: {csv_file}")
            continue
        
        df = pd.read_csv(csv_file)
        df['dataset_name'] = dataset_name
        all_data.append(df)
        print(f"✅ Loaded {len(df)} results from {dataset_name}")
    
    if not all_data:
        print("❌ No results found!")
        return None
    
    return pd.concat(all_data, ignore_index=True)

def analyze_method_ranking(df):
    """Analyze method ranking across datasets"""
    print("\n" + "="*80)
    print("METHOD RANKING ACROSS DATASETS")
    print("="*80)
    
    for component in ['S', 'T']:
        comp_name = "Seasonal" if component == 'S' else "Trend"
        print(f"\n{comp_name} Component:")
        print("-"*40)
        
        df_comp = df[df['ablation'] == component]
        
        # Average MSE by method across all datasets and horizons
        ranking = df_comp.groupby('decomp')['MSE'].mean().sort_values()
        
        print("\nOverall Ranking (Average MSE across all datasets):")
        for rank, (method, mse) in enumerate(ranking.items(), 1):
            print(f"  {rank}. {method:10s}: {mse:.4f}")
        
        # Per-dataset ranking
        print(f"\nPer-Dataset Best Method:")
        for dataset in df_comp['dataset_name'].unique():
            df_ds = df_comp[df_comp['dataset_name'] == dataset]
            best_method = df_ds.groupby('decomp')['MSE'].mean().idxmin()
            best_mse = df_ds.groupby('decomp')['MSE'].mean().min()
            print(f"  {dataset:10s}: {best_method:10s} (MSE: {best_mse:.4f})")

def analyze_stability(df):
    """Analyze method stability across datasets"""
    print("\n" + "="*80)
    print("METHOD STABILITY ANALYSIS")
    print("="*80)
    
    for component in ['S', 'T']:
        comp_name = "Seasonal" if component == 'S' else "Trend"
        print(f"\n{comp_name} Component:")
        print("-"*40)
        
        df_comp = df[df['ablation'] == component]
        
        # Calculate mean and std of MSE for each method
        stability = df_comp.groupby('decomp')['MSE'].agg(['mean', 'std', 'min', 'max'])
        stability['cv'] = stability['std'] / stability['mean']  # Coefficient of variation
        stability = stability.sort_values('cv')
        
        print("\nMethod Stability (Lower CV = More Stable):")
        print(stability.to_string())

def generate_cross_dataset_heatmap(df, output_dir="outputs/decomp_linear_bench/cross_dataset/"):
    """Generate cross-dataset comparison heatmap"""
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)
    
    for component in ['S', 'T']:
        comp_name = "Seasonal" if component == 'S' else "Trend"
        df_comp = df[df['ablation'] == component]
        
        # Pivot: methods × datasets
        pivot = df_comp.groupby(['decomp', 'dataset_name'])['MSE'].mean().unstack()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(pivot, annot=True, fmt='.4f', cmap='RdYlGn_r', ax=ax,
                   cbar_kws={'label': 'MSE'})
        ax.set_title(f'{comp_name} Component Performance Across Datasets',
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Dataset', fontsize=12)
        ax.set_ylabel('Method', fontsize=12)
        
        out_file = os.path.join(output_dir, f"cross_dataset_{component}_heatmap.png")
        plt.tight_layout()
        plt.savefig(out_file, dpi=300, bbox_inches='tight')
        print(f"  Saved: {out_file}")
        plt.close()

def generate_summary_report(df, output_dir="outputs/decomp_linear_bench/cross_dataset/"):
    """Generate summary report"""
    os.makedirs(output_dir, exist_ok=True)
    
    report_lines = []
    report_lines.append("="*80)
    report_lines.append("CROSS-DATASET DECOMPOSITION METHOD COMPARISON")
    report_lines.append("="*80)
    report_lines.append(f"Datasets: {', '.join(df['dataset_name'].unique())}")
    report_lines.append(f"Methods: {', '.join(df['decomp'].unique())}")
    report_lines.append(f"Total experiments: {len(df)}")
    report_lines.append("="*80)
    
    # Overall winners
    report_lines.append("\n" + "="*80)
    report_lines.append("OVERALL WINNERS")
    report_lines.append("="*80)
    
    for component in ['S', 'T']:
        comp_name = "Seasonal" if component == 'S' else "Trend"
        df_comp = df[df['ablation'] == component]
        
        # Overall best
        overall_ranking = df_comp.groupby('decomp')['MSE'].mean().sort_values()
        best_method = overall_ranking.index[0]
        best_mse = overall_ranking.values[0]
        
        report_lines.append(f"\nBest {comp_name} Component (Overall): {best_method}")
        report_lines.append(f"  Average MSE across all datasets: {best_mse:.4f}")
        
        # Dataset-specific winners
        report_lines.append(f"\n  Dataset-specific winners:")
        for dataset in df_comp['dataset_name'].unique():
            df_ds = df_comp[df_comp['dataset_name'] == dataset]
            ds_best = df_ds.groupby('decomp')['MSE'].mean().idxmin()
            ds_mse = df_ds.groupby('decomp')['MSE'].mean().min()
            report_lines.append(f"    {dataset:10s}: {ds_best:10s} (MSE: {ds_mse:.4f})")
    
    # Recommendations
    report_lines.append("\n" + "="*80)
    report_lines.append("RECOMMENDATIONS")
    report_lines.append("="*80)
    
    report_lines.append("\n1. Universal Method:")
    # Find method with lowest average rank
    ranks = []
    for dataset in df['dataset_name'].unique():
        for component in ['S', 'T']:
            df_subset = df[(df['dataset_name'] == dataset) & (df['ablation'] == component)]
            method_ranks = df_subset.groupby('decomp')['MSE'].mean().rank()
            ranks.append(method_ranks)
    
    avg_ranks = pd.concat(ranks, axis=1).mean(axis=1).sort_values()
    universal_best = avg_ranks.index[0]
    report_lines.append(f"   {universal_best} - Most consistent across all datasets and components")
    
    report_lines.append("\n2. Component-Specific:")
    for component in ['S', 'T']:
        comp_name = "Seasonal" if component == 'S' else "Trend"
        df_comp = df[df['ablation'] == component]
        best_method = df_comp.groupby('decomp')['MSE'].mean().idxmin()
        report_lines.append(f"   {comp_name}: {best_method}")
    
    report_lines.append("\n3. Avoid:")
    # Find consistently worst performers
    worst_methods = avg_ranks.tail(2)
    for method in worst_methods.index:
        report_lines.append(f"   {method} - Consistently underperforms")
    
    report_text = "\n".join(report_lines)
    
    out_file = os.path.join(output_dir, "cross_dataset_summary.txt")
    with open(out_file, 'w') as f:
        f.write(report_text)
    
    print(report_text)
    print(f"\n  Report saved: {out_file}")

def main():
    print("="*80)
    print("CROSS-DATASET ANALYSIS")
    print("="*80)
    
    # Load all results
    df = load_all_results()
    
    if df is None or len(df) == 0:
        print("❌ No results to analyze!")
        return
    
    print(f"\n✅ Loaded {len(df)} total experiments from {df['dataset_name'].nunique()} datasets")
    
    # Analysis
    analyze_method_ranking(df)
    analyze_stability(df)
    generate_cross_dataset_heatmap(df)
    generate_summary_report(df)
    
    print("\n" + "="*80)
    print("CROSS-DATASET ANALYSIS COMPLETE!")
    print("="*80)

if __name__ == "__main__":
    main()
