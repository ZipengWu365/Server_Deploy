#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Cross-Method Component Comparison
Compare Seasonal and Trend components from different decomposition methods
"""

import os
import sys
import yaml
import pandas as pd
import numpy as np
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from features.decomp_linear_bench.runner import run_experiment

def main():
    parser = argparse.ArgumentParser(description="Run component comparison experiments")
    parser.add_argument('--config', type=str, default='configs/decomp_linear/compare_components.yaml',
                       help='Path to config YAML file')
    args = parser.parse_args()
    
    config_path = args.config
    
    print("="*80)
    print("Cross-Method Component Comparison")
    print("="*80)
    print(f"Config: {config_path}")
    print()
    
    if not os.path.exists(config_path):
        print(f"Error: Config file not found: {config_path}")
        return
    
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    
    print("Configuration:")
    print(f"  Dataset: {cfg['dataset']['name']}")
    print(f"  Methods: {cfg['decomp']}")
    print(f"  Ablations: {cfg['ablation_modes']}")
    print(f"  Horizons: {cfg['horizons']}")
    print(f"  Learner: {cfg['learner']}")
    print()
    
    # Run experiment
    try:
        run_experiment(cfg)
        print("\n" + "="*80)
        print("Experiment Complete!")
        print("="*80)
        
        # Check output
        out_dir = cfg.get("out_dir", "outputs/decomp_linear_bench/compare_components/")
        summary_file = os.path.join(out_dir, "metrics_summary_by_method_horizon_ablation.csv")
        
        if os.path.exists(summary_file):
            df = pd.read_csv(summary_file)
            print(f"\nResults saved to: {summary_file}")
            print(f"Total experiments: {len(df)}")
            print("\nPreview:")
            print(df.head(10))
        else:
            print(f"\nWarning: Summary file not found at {summary_file}")
            
    except Exception as e:
        print(f"\nError during experiment: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
