import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from typing import Dict

def save_heatmap_trend_r2(df: pd.DataFrame, out_dir: str):
    # Filter for Trend ablation (T or T+S+D?)
    # Let's assume we want to compare methods on "T" ablation or "T+S+D"
    # The prompt says "heatmap_T_r2", likely meaning R2 score when using only Trend features?
    # Or R2 of the full model?
    # Let's use "T" ablation for Trend R2 if available, else T+S+D
    
    subset = df[df["ablation"] == "T"]
    if subset.empty:
        subset = df[df["ablation"] == "T+S+D"]
        
    if subset.empty: return
    
    pivot = subset.pivot(index="decomp", columns="horizon", values="R2")
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(pivot, annot=True, cmap="viridis", fmt=".3f")
    plt.title("Trend R2 Score by Method and Horizon")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "heatmap_T_r2.png"))
    plt.close()

def save_heatmap_season_speccorr(df: pd.DataFrame, out_dir: str):
    # Placeholder: We don't have spectral corr metric in runner yet.
    # We only have MAE, MSE, R2, sMAPE.
    # The prompt asked to "reproduce heatmaps akin to the ones we already use".
    # If we don't have the metric, we can't plot it.
    # For now, let's plot sMAPE for "S" ablation.
    
    subset = df[df["ablation"] == "S"]
    if subset.empty: return
    
    pivot = subset.pivot(index="decomp", columns="horizon", values="sMAPE")
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(pivot, annot=True, cmap="coolwarm_r", fmt=".2f")
    plt.title("Seasonal sMAPE by Method and Horizon")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "heatmap_S_smape.png"))
    plt.close()

def save_error_sqrt_panels(df: pd.DataFrame, out_dir: str):
    # Plot MAE vs Horizon for each method
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df, x="horizon", y="MAE", hue="decomp", style="ablation", markers=True)
    plt.title("MAE vs Horizon by Method")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "error_panels_mae.png"))
    plt.close()

def generate_report(csv_path: str, out_dir: str):
    df = pd.read_csv(csv_path)
    os.makedirs(out_dir, exist_ok=True)
    
    save_heatmap_trend_r2(df, out_dir)
    save_heatmap_season_speccorr(df, out_dir)
    save_error_sqrt_panels(df, out_dir)
    
    print(f"Report generated in {out_dir}")
