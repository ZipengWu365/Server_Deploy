#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Run all dataset experiments sequentially
"""

import os
import sys
import yaml
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features.decomp_linear_bench.runner import run_experiment

# Configuration files
CONFIGS = [
    ("ETTh1 (All Methods)", "configs/decomp_linear/etth1_all.yaml"),
    ("ETTh2 (All Methods)", "configs/decomp_linear/etth2_all.yaml"),
    ("ETTm1 (Core Methods)", "configs/decomp_linear/ettm1_all.yaml"),
    ("ETTm2 (Core Methods)", "configs/decomp_linear/ettm2_all.yaml"),
    ("Exchange Rate (Core Methods)", "configs/decomp_linear/exchange_all.yaml"),
]

def run_single_config(name, config_path):
    """Run single experiment configuration"""
    print("\n" + "="*80)
    print(f"Starting: {name}")
    print(f"Config: {config_path}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    if not os.path.exists(config_path):
        print(f"❌ Config file not found: {config_path}")
        return False
    
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    
    print(f"Dataset: {cfg['dataset']['name']}")
    print(f"Methods: {cfg['decomp']}")
    print(f"Ablations: {cfg['ablation_modes']}")
    print(f"Horizons: {cfg['horizons']}")
    print()
    
    start_time = time.time()
    
    try:
        run_experiment(cfg)
        elapsed = time.time() - start_time
        print(f"\n✅ {name} completed in {elapsed/60:.1f} minutes")
        return True
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\n❌ {name} failed after {elapsed/60:.1f} minutes")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("="*80)
    print("BATCH EXPERIMENT RUNNER - ALL DATASETS")
    print("="*80)
    print(f"Total configurations: {len(CONFIGS)}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    results = []
    total_start = time.time()
    
    for i, (name, config_path) in enumerate(CONFIGS, 1):
        print(f"\n[{i}/{len(CONFIGS)}] Processing: {name}")
        success = run_single_config(name, config_path)
        results.append((name, success))
        
        if i < len(CONFIGS):
            print(f"\n⏳ Moving to next dataset...")
            time.sleep(2)
    
    # Summary
    total_elapsed = time.time() - total_start
    
    print("\n" + "="*80)
    print("BATCH EXPERIMENT SUMMARY")
    print("="*80)
    print(f"Total time: {total_elapsed/60:.1f} minutes ({total_elapsed/3600:.2f} hours)")
    print(f"\nResults:")
    
    success_count = 0
    for name, success in results:
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"  {status}: {name}")
        if success:
            success_count += 1
    
    print(f"\nSuccess rate: {success_count}/{len(CONFIGS)} ({success_count/len(CONFIGS)*100:.1f}%)")
    print("="*80)
    
    # Generate cross-dataset analysis
    if success_count > 0:
        print("\n🔍 Generating cross-dataset analysis...")
        try:
            os.system("python analyze_cross_dataset.py")
        except:
            print("⚠️ Cross-dataset analysis skipped (script not found)")
    
    print("\n🎉 All done!")

if __name__ == "__main__":
    main()
