"""
Analyze and summarize decomposition benchmark results.
Compares all methods against baseline and generates summary reports.
"""
import os
import pandas as pd
import numpy as np
from pathlib import Path

def load_all_results(base_dir="outputs/decomp_linear_bench"):
    """Load all experiment results from baseline and all_methods directories."""
    all_dfs = []
    
    # Load baseline results
    baseline_dir = os.path.join(base_dir, "baseline")
    if os.path.exists(baseline_dir):
        for dataset_dir in os.listdir(baseline_dir):
            dataset_path = os.path.join(baseline_dir, dataset_dir)
            if os.path.isdir(dataset_path):
                csv_path = os.path.join(dataset_path, "metrics_summary_by_method_horizon_ablation.csv")
                if os.path.exists(csv_path):
                    df = pd.read_csv(csv_path)
                    df['source'] = 'baseline'
                    all_dfs.append(df)
                    print(f"Loaded baseline: {dataset_dir}")
    
    # Load all_methods results
    methods_dir = os.path.join(base_dir, "all_methods")
    if os.path.exists(methods_dir):
        for dataset_dir in os.listdir(methods_dir):
            dataset_path = os.path.join(methods_dir, dataset_dir)
            if os.path.isdir(dataset_path):
                csv_path = os.path.join(dataset_path, "metrics_summary_by_method_horizon_ablation.csv")
                if os.path.exists(csv_path):
                    df = pd.read_csv(csv_path)
                    df['source'] = 'all_methods'
                    all_dfs.append(df)
                    print(f"Loaded all_methods: {dataset_dir}")
    
    if not all_dfs:
        print("No results found!")
        return None
    
    # Combine all results
    combined = pd.concat(all_dfs, ignore_index=True)
    return combined

def calculate_improvements(df):
    """Calculate improvements relative to baseline."""
    results = []
    
    datasets = df['dataset'].unique()
    
    for dataset in datasets:
        df_dataset = df[df['dataset'] == dataset]
        
        # Get baseline results
        baseline = df_dataset[(df_dataset['decomp'] == 'NONE') & (df_dataset['ablation'] == 'RAW')]
        
        if baseline.empty:
            print(f"Warning: No baseline found for {dataset}")
            continue
        
        # Compare each method/ablation against baseline
        methods = df_dataset[(df_dataset['decomp'] != 'NONE') | (df_dataset['ablation'] != 'RAW')]
        
        for _, row in methods.iterrows():
            horizon = row['horizon']
            baseline_row = baseline[baseline['horizon'] == horizon]
            
            if baseline_row.empty:
                continue
            
            baseline_mae = baseline_row.iloc[0]['MAE']
            baseline_mse = baseline_row.iloc[0]['MSE']
            
            mae_improvement = ((baseline_mae - row['MAE']) / baseline_mae) * 100
            mse_improvement = ((baseline_mse - row['MSE']) / baseline_mse) * 100
            
            results.append({
                'dataset': dataset,
                'decomp': row['decomp'],
                'ablation': row['ablation'],
                'horizon': horizon,
                'MAE': row['MAE'],
                'MSE': row['MSE'],
                'MAPE': row['MAPE'],
                'baseline_MAE': baseline_mae,
                'baseline_MSE': baseline_mse,
                'MAE_improvement_%': mae_improvement,
                'MSE_improvement_%': mse_improvement,
                'n_feat': row['n_feat']
            })
    
    return pd.DataFrame(results)

def generate_summary_tables(df_improvements, output_dir="outputs/decomp_linear_bench/analysis"):
    """Generate summary tables and statistics."""
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Overall summary: Average performance by method and ablation
    summary_by_method = df_improvements.groupby(['decomp', 'ablation']).agg({
        'MAE': 'mean',
        'MAE_improvement_%': 'mean',
        'MSE_improvement_%': 'mean'
    }).round(4)
    summary_by_method = summary_by_method.sort_values('MAE_improvement_%', ascending=False)
    
    print("\n" + "="*80)
    print("Summary by Method and Ablation (Average across all datasets and horizons)")
    print("="*80)
    print(summary_by_method)
    
    summary_path = os.path.join(output_dir, "summary_by_method_ablation.csv")
    summary_by_method.to_csv(summary_path)
    print(f"\nSaved: {summary_path}")
    
    # 2. Best method per dataset
    best_by_dataset = df_improvements.loc[
        df_improvements.groupby(['dataset', 'horizon'])['MAE'].idxmin()
    ][['dataset', 'horizon', 'decomp', 'ablation', 'MAE', 'MAE_improvement_%']]
    
    print("\n" + "="*80)
    print("Best Method per Dataset and Horizon")
    print("="*80)
    print(best_by_dataset.to_string(index=False))
    
    best_path = os.path.join(output_dir, "best_method_per_dataset_horizon.csv")
    best_by_dataset.to_csv(best_path, index=False)
    print(f"\nSaved: {best_path}")
    
    # 3. Detailed results by dataset
    for dataset in df_improvements['dataset'].unique():
        df_dataset = df_improvements[df_improvements['dataset'] == dataset]
        dataset_path = os.path.join(output_dir, f"detailed_{dataset.lower()}.csv")
        df_dataset.to_csv(dataset_path, index=False)
        print(f"Saved detailed results: {dataset_path}")
    
    # 4. Component comparison: +T vs +S
    component_comparison = df_improvements[df_improvements['ablation'].isin(['+T', '+S'])].groupby(
        ['decomp', 'ablation']
    ).agg({
        'MAE_improvement_%': ['mean', 'std', 'count']
    }).round(4)
    
    print("\n" + "="*80)
    print("Component Comparison: +T (Trend) vs +S (Seasonal)")
    print("="*80)
    print(component_comparison)
    
    comp_path = os.path.join(output_dir, "component_comparison.csv")
    component_comparison.to_csv(comp_path)
    print(f"\nSaved: {comp_path}")
    
    # 5. Horizon analysis: How performance changes with prediction length
    horizon_analysis = df_improvements.groupby(['horizon', 'ablation']).agg({
        'MAE_improvement_%': 'mean'
    }).round(4)
    
    print("\n" + "="*80)
    print("Horizon Analysis: Performance vs Prediction Length")
    print("="*80)
    print(horizon_analysis)
    
    horizon_path = os.path.join(output_dir, "horizon_analysis.csv")
    horizon_analysis.to_csv(horizon_path)
    print(f"\nSaved: {horizon_path}")
    
    # 6. Full comparison table
    full_path = os.path.join(output_dir, "full_comparison.csv")
    df_improvements.to_csv(full_path, index=False)
    print(f"\nSaved full comparison: {full_path}")
    
    return summary_by_method, best_by_dataset

def generate_markdown_report(df_improvements, output_dir="outputs/decomp_linear_bench/analysis"):
    """Generate a markdown report summarizing findings."""
    os.makedirs(output_dir, exist_ok=True)
    
    report_path = os.path.join(output_dir, "RESULTS_SUMMARY.md")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Ridge Regression + Decomposition Components: Experimental Results\n\n")
        f.write(f"Generated: {pd.Timestamp.now()}\n\n")
        
        # Overall findings
        f.write("## Overall Findings\n\n")
        
        avg_improvements = df_improvements.groupby(['decomp', 'ablation'])['MAE_improvement_%'].mean()
        best_method = avg_improvements.idxmax()
        best_improvement = avg_improvements.max()
        
        f.write(f"**Best Overall Method**: {best_method[0]} with {best_method[1]} ablation\n")
        f.write(f"**Average MAE Improvement**: {best_improvement:.2f}%\n\n")
        
        # Trend vs Seasonal
        f.write("## Component Analysis: Trend (+T) vs Seasonal (+S)\n\n")
        
        trend_avg = df_improvements[df_improvements['ablation'] == '+T']['MAE_improvement_%'].mean()
        seasonal_avg = df_improvements[df_improvements['ablation'] == '+S']['MAE_improvement_%'].mean()
        
        f.write(f"- **Trend (+T) Average Improvement**: {trend_avg:.2f}%\n")
        f.write(f"- **Seasonal (+S) Average Improvement**: {seasonal_avg:.2f}%\n\n")
        
        if trend_avg > seasonal_avg:
            f.write("✅ **Conclusion**: Trend components generally provide better improvements.\n\n")
        else:
            f.write("✅ **Conclusion**: Seasonal components generally provide better improvements.\n\n")
        
        # By dataset
        f.write("## Results by Dataset\n\n")
        
        for dataset in sorted(df_improvements['dataset'].unique()):
            f.write(f"### {dataset}\n\n")
            
            df_dataset = df_improvements[df_improvements['dataset'] == dataset]
            best_row = df_dataset.loc[df_dataset['MAE'].idxmin()]
            
            f.write(f"**Best Method**: {best_row['decomp']} {best_row['ablation']}\n")
            f.write(f"**Best MAE**: {best_row['MAE']:.4f} (Improvement: {best_row['MAE_improvement_%']:.2f}%)\n")
            f.write(f"**Horizon**: {best_row['horizon']}\n\n")
            
            # Top 3 methods for this dataset
            top3 = df_dataset.nsmallest(3, 'MAE')[['decomp', 'ablation', 'horizon', 'MAE', 'MAE_improvement_%']]
            f.write("**Top 3 Configurations**:\n\n")
            f.write(top3.to_markdown(index=False))
            f.write("\n\n")
        
        # Horizon trends
        f.write("## Horizon Analysis\n\n")
        
        f.write("Average MAE improvement by prediction horizon:\n\n")
        horizon_summary = df_improvements.groupby('horizon')['MAE_improvement_%'].mean().round(2)
        for horizon, improvement in horizon_summary.items():
            f.write(f"- **H={horizon}**: {improvement:.2f}%\n")
        f.write("\n")
        
        # Method ranking
        f.write("## Method Ranking (by average MAE improvement)\n\n")
        
        method_ranking = df_improvements.groupby('decomp').agg({
            'MAE_improvement_%': 'mean',
            'MAE': 'mean'
        }).sort_values('MAE_improvement_%', ascending=False).round(4)
        
        f.write(method_ranking.to_markdown())
        f.write("\n\n")
        
        f.write("## Files Generated\n\n")
        f.write("- `summary_by_method_ablation.csv`: Average performance by method/ablation\n")
        f.write("- `best_method_per_dataset_horizon.csv`: Best configuration for each dataset/horizon\n")
        f.write("- `component_comparison.csv`: Trend vs Seasonal comparison\n")
        f.write("- `horizon_analysis.csv`: Performance by prediction length\n")
        f.write("- `full_comparison.csv`: Complete results table\n")
        f.write("- `detailed_<dataset>.csv`: Per-dataset detailed results\n")
    
    print(f"\n{'='*80}")
    print(f"Markdown report saved: {report_path}")
    print(f"{'='*80}")

def main():
    print("="*80)
    print("Decomposition Linear Benchmark - Results Analysis")
    print("="*80)
    
    # Load all results
    df = load_all_results()
    
    if df is None:
        return
    
    print(f"\nTotal records loaded: {len(df)}")
    print(f"Datasets: {df['dataset'].unique()}")
    print(f"Methods: {df['decomp'].unique()}")
    print(f"Ablations: {df['ablation'].unique()}")
    
    # Calculate improvements
    print("\nCalculating improvements relative to baseline...")
    df_improvements = calculate_improvements(df)
    
    print(f"Improvement records: {len(df_improvements)}")
    
    # Generate summary tables
    generate_summary_tables(df_improvements)
    
    # Generate markdown report
    generate_markdown_report(df_improvements)
    
    print("\n" + "="*80)
    print("Analysis complete!")
    print("="*80)

if __name__ == "__main__":
    main()
