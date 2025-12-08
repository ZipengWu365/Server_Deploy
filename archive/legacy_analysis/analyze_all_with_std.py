"""
Comprehensive analysis combining:
1. STD series ablation results (T, S, D, T+S+D from STD_MULTI and STD_FULL)
2. Cross-method component comparison (STL-S, STL-T, SSA-S, SSA-T, etc.)
"""
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)

def load_all_results():
    """Load all experimental results"""
    base_dir = Path("outputs/decomp_linear_bench")
    
    # 1. Load cross-method results (current experiments)
    cross_method_dfs = []
    for dataset_dir in ["etth1_all_methods", "etth2_all_methods", "ettm1_all_methods", 
                        "ettm2_all_methods", "exchange_all_methods"]:
        csv_path = base_dir / dataset_dir / "metrics_summary_by_method_horizon_ablation.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            cross_method_dfs.append(df)
            print(f"✓ Loaded: {dataset_dir}")
        else:
            print(f"✗ Missing: {dataset_dir}")
    
    # 2. Load STD series results
    std_dfs = []
    for std_dir in ["quick_std", "debug_std"]:
        csv_path = base_dir / std_dir / "metrics_summary_by_method_horizon_ablation.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            std_dfs.append(df)
            print(f"✓ Loaded STD: {std_dir}")
    
    cross_df = pd.concat(cross_method_dfs, ignore_index=True) if cross_method_dfs else pd.DataFrame()
    std_df = pd.concat(std_dfs, ignore_index=True) if std_dfs else pd.DataFrame()
    
    return cross_df, std_df

def create_unified_comparison(cross_df, std_df):
    """
    Create unified comparison table:
    - STD_MULTI-T, STD_MULTI-S (multi-scale seasonal/trend)
    - STD_FULL-T, STD_FULL-S (full decomposition seasonal/trend)
    - STL-T, STL-S, SSA-T, SSA-S, etc. (single-scale methods)
    """
    results = []
    
    # Process cross-method results (STL, SSA, etc.)
    if not cross_df.empty:
        for _, row in cross_df.iterrows():
            method_comp = f"{row['decomp']}-{row['ablation']}"
            results.append({
                'Dataset': row['dataset'],
                'Method': method_comp,
                'Horizon': row['horizon'],
                'Features': row['n_feat'],
                'MSE': row['MSE'],
                'MAE': row['MAE'],
                'R2': row['R2'],
                'Category': 'Single-Scale'
            })
    
    # Process STD results
    if not std_df.empty:
        for _, row in std_df.iterrows():
            # Map ablation codes
            if row['ablation'] == 'T':
                method_name = f"{row['decomp']}-Trend"
            elif row['ablation'] == 'S':
                method_name = f"{row['decomp']}-Season"
            elif row['ablation'] == 'D':
                method_name = f"{row['decomp']}-Detail"
            elif row['ablation'] == 'T+S+D':
                method_name = f"{row['decomp']}-Full"
            else:
                method_name = f"{row['decomp']}-{row['ablation']}"
            
            category = 'Multi-Scale' if 'MULTI' in row['decomp'] else 'Full-Decomp'
            
            results.append({
                'Dataset': row['dataset'],
                'Method': method_name,
                'Horizon': row['horizon'],
                'Features': row['n_feat'],
                'MSE': row['MSE'],
                'MAE': row['MAE'],
                'R2': row['R2'],
                'Category': category
            })
    
    unified_df = pd.DataFrame(results)
    return unified_df

def analyze_by_component_type(unified_df):
    """Compare seasonal vs trend components across all methods"""
    
    # Extract component type (Seasonal vs Trend)
    def get_component_type(method):
        if any(x in method for x in ['-S', 'Season']):
            return 'Seasonal'
        elif any(x in method for x in ['-T', 'Trend']):
            return 'Trend'
        elif '-D' in method or 'Detail' in method:
            return 'Detail'
        else:
            return 'Combined'
    
    unified_df['Component'] = unified_df['Method'].apply(get_component_type)
    
    # 1. Overall ranking by component type
    print("\n" + "="*80)
    print("COMPONENT TYPE COMPARISON (Average MSE across all datasets & horizons)")
    print("="*80)
    
    component_stats = unified_df.groupby('Component').agg({
        'MSE': ['mean', 'std', 'min'],
        'MAE': ['mean', 'std'],
        'R2': ['mean', 'std']
    }).round(4)
    print(component_stats)
    
    # 2. Method ranking within each component type
    print("\n" + "="*80)
    print("TOP METHODS FOR SEASONAL COMPONENTS")
    print("="*80)
    seasonal_methods = unified_df[unified_df['Component'] == 'Seasonal'].groupby('Method').agg({
        'MSE': 'mean',
        'MAE': 'mean',
        'R2': 'mean'
    }).sort_values('MSE')
    print(seasonal_methods.head(10))
    
    print("\n" + "="*80)
    print("TOP METHODS FOR TREND COMPONENTS")
    print("="*80)
    trend_methods = unified_df[unified_df['Component'] == 'Trend'].groupby('Method').agg({
        'MSE': 'mean',
        'MAE': 'mean',
        'R2': 'mean'
    }).sort_values('MSE')
    print(trend_methods.head(10))
    
    return component_stats

def compare_std_vs_others(unified_df):
    """Compare STD multi-scale vs single-scale methods"""
    print("\n" + "="*80)
    print("MULTI-SCALE (STD) vs SINGLE-SCALE METHODS")
    print("="*80)
    
    category_comparison = unified_df.groupby('Category').agg({
        'MSE': ['mean', 'std', 'min'],
        'MAE': ['mean', 'std'],
        'Features': 'mean'
    }).round(4)
    print(category_comparison)
    
    # Specific comparison: STD_MULTI-Season vs STL-S, SSA-S, etc.
    print("\n" + "="*80)
    print("SEASONAL COMPONENT DETAILED COMPARISON")
    print("="*80)
    
    seasonal_df = unified_df[unified_df['Component'] == 'Seasonal']
    seasonal_avg = seasonal_df.groupby(['Method', 'Category']).agg({
        'MSE': 'mean',
        'MAE': 'mean',
        'R2': 'mean'
    }).sort_values('MSE')
    print(seasonal_avg)
    
    print("\n" + "="*80)
    print("TREND COMPONENT DETAILED COMPARISON")
    print("="*80)
    
    trend_df = unified_df[unified_df['Component'] == 'Trend']
    trend_avg = trend_df.groupby(['Method', 'Category']).agg({
        'MSE': 'mean',
        'MAE': 'mean',
        'R2': 'mean'
    }).sort_values('MSE')
    print(trend_avg)

def generate_visualizations(unified_df):
    """Generate comparison plots"""
    output_dir = Path("outputs/decomp_linear_bench/unified_analysis")
    output_dir.mkdir(exist_ok=True)
    
    # 1. Component type comparison boxplot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for idx, metric in enumerate(['MSE', 'MAE', 'R2']):
        ax = axes[idx]
        unified_df.boxplot(column=metric, by='Component', ax=ax)
        ax.set_title(f'{metric} by Component Type')
        ax.set_xlabel('Component Type')
        ax.set_ylabel(metric)
    
    plt.suptitle('Component Type Performance Comparison')
    plt.tight_layout()
    plt.savefig(output_dir / 'component_type_comparison.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved: {output_dir / 'component_type_comparison.png'}")
    
    # 2. Top methods heatmap
    plt.figure(figsize=(12, 8))
    
    # Get top 15 methods by average MSE
    top_methods = unified_df.groupby('Method')['MSE'].mean().sort_values().head(15)
    
    # Create pivot table for heatmap
    heatmap_data = unified_df[unified_df['Method'].isin(top_methods.index)].pivot_table(
        index='Method',
        columns='Horizon',
        values='MSE',
        aggfunc='mean'
    )
    
    sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='RdYlGn_r', cbar_kws={'label': 'MSE'})
    plt.title('Top 15 Methods: MSE across Horizons')
    plt.xlabel('Prediction Horizon')
    plt.ylabel('Method')
    plt.tight_layout()
    plt.savefig(output_dir / 'top_methods_heatmap.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir / 'top_methods_heatmap.png'}")
    
    # 3. Category comparison (Multi-scale vs Single-scale)
    plt.figure(figsize=(10, 6))
    
    category_data = unified_df.groupby(['Category', 'Horizon'])['MSE'].mean().reset_index()
    
    for category in category_data['Category'].unique():
        data = category_data[category_data['Category'] == category]
        plt.plot(data['Horizon'], data['MSE'], marker='o', label=category, linewidth=2)
    
    plt.xlabel('Prediction Horizon', fontsize=12)
    plt.ylabel('Average MSE', fontsize=12)
    plt.title('Multi-Scale vs Single-Scale Performance', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'category_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir / 'category_comparison.png'}")

def save_summary_tables(unified_df):
    """Save summary tables to CSV"""
    output_dir = Path("outputs/decomp_linear_bench/unified_analysis")
    output_dir.mkdir(exist_ok=True)
    
    # 1. Full unified results
    unified_df.to_csv(output_dir / 'unified_all_results.csv', index=False)
    print(f"\n✓ Saved: {output_dir / 'unified_all_results.csv'}")
    
    # 2. Method ranking table
    method_ranking = unified_df.groupby('Method').agg({
        'MSE': ['mean', 'std', 'min'],
        'MAE': ['mean', 'std'],
        'R2': ['mean', 'std'],
        'Features': 'mean',
        'Category': 'first'
    }).round(4)
    method_ranking.columns = ['_'.join(col).strip() for col in method_ranking.columns]
    method_ranking = method_ranking.sort_values('MSE_mean')
    method_ranking.to_csv(output_dir / 'method_ranking.csv')
    print(f"✓ Saved: {output_dir / 'method_ranking.csv'}")
    
    # 3. Component type summary
    component_summary = unified_df.copy()
    component_summary['Component'] = component_summary['Method'].apply(
        lambda x: 'Seasonal' if any(s in x for s in ['-S', 'Season']) 
        else ('Trend' if any(t in x for t in ['-T', 'Trend']) 
        else ('Detail' if '-D' in x or 'Detail' in x else 'Combined'))
    )
    
    component_stats = component_summary.groupby(['Component', 'Category']).agg({
        'MSE': ['mean', 'std', 'count'],
        'MAE': 'mean',
        'R2': 'mean'
    }).round(4)
    component_stats.to_csv(output_dir / 'component_category_summary.csv')
    print(f"✓ Saved: {output_dir / 'component_category_summary.csv'}")

def main():
    print("\n" + "="*80)
    print("UNIFIED ANALYSIS: STD Multi-Scale vs Cross-Method Components")
    print("="*80)
    
    # Load all results
    cross_df, std_df = load_all_results()
    
    if cross_df.empty and std_df.empty:
        print("\n✗ No results found!")
        return
    
    print(f"\nCross-method results: {len(cross_df)} rows")
    print(f"STD results: {len(std_df)} rows")
    
    # Create unified comparison
    unified_df = create_unified_comparison(cross_df, std_df)
    print(f"\nUnified dataset: {len(unified_df)} rows")
    
    # Analysis
    analyze_by_component_type(unified_df)
    compare_std_vs_others(unified_df)
    
    # Visualizations
    generate_visualizations(unified_df)
    
    # Save tables
    save_summary_tables(unified_df)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print("\nKey Findings:")
    print("1. Check outputs/decomp_linear_bench/unified_analysis/ for all results")
    print("2. Compare Multi-Scale (STD) vs Single-Scale methods")
    print("3. Identify best Seasonal vs Trend components")

if __name__ == "__main__":
    main()
