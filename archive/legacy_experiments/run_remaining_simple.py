"""
Simple sequential runner for remaining datasets to avoid KeyboardInterrupt
"""
import yaml
import pandas as pd
from pathlib import Path
from features.decomp_linear_bench.runner import run_experiment

def run_simple(config_path):
    """Run experiment without parallel processing"""
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    # Force sequential execution
    cfg['n_jobs'] = 1
    
    print(f"\n{'='*80}")
    print(f"Running: {config_path}")
    print(f"Dataset: {cfg['dataset']['name']}")
    print(f"Methods: {cfg['decomp']}")
    print(f"{'='*80}\n")
    
    try:
        run_experiment(cfg)
        print(f"\n✓ Completed: {cfg['dataset']['name']}")
    except Exception as e:
        print(f"\n✗ Failed: {cfg['dataset']['name']}")
        print(f"Error: {e}")

if __name__ == "__main__":
    configs = [
        "configs/decomp_linear/ettm1_all.yaml",
        "configs/decomp_linear/ettm2_all.yaml", 
        "configs/decomp_linear/exchange_all.yaml"
    ]
    
    for cfg_path in configs:
        run_simple(cfg_path)
