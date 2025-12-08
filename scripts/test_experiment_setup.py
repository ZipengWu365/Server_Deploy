"""
Quick test script to verify the experiment setup works correctly.
Runs a minimal test on ETTh1 with reduced horizons.
"""
import yaml
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from features.decomp_linear_bench.runner import run_experiment

def test_baseline():
    """Test baseline experiment."""
    print("="*80)
    print("Testing Baseline Experiment")
    print("="*80)
    
    cfg = {
        "dataset": {
            "name": "ETTh1",
            "path": "dataset/ETTh1.csv",
            "split": [8545, 2881, 2881]
        },
        "lookback": 336,
        "horizons": [96],  # Only test one horizon
        "decomp": [],
        "ablation_modes": ["RAW"],
        "learner": "Ridge",
        "learner_params": {"alpha": 1.0},
        "seed": 42,
        "out_dir": "outputs/decomp_linear_bench/test/baseline/"
    }
    
    try:
        result = run_experiment(cfg)
        print("\n✅ Baseline test passed!")
        print(result)
        return True
    except Exception as e:
        print(f"\n❌ Baseline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_decomposition():
    """Test decomposition experiment with one method."""
    print("\n" + "="*80)
    print("Testing Decomposition Experiment (STL)")
    print("="*80)
    
    cfg = {
        "dataset": {
            "name": "ETTh1",
            "path": "dataset/ETTh1.csv",
            "split": [8545, 2881, 2881]
        },
        "lookback": 336,
        "horizons": [96],  # Only test one horizon
        "decomp": [
            {"method": "STL", "params": {"period": 24}}
        ],
        "ablation_modes": ["+T"],  # Only test trend
        "learner": "Ridge",
        "learner_params": {"alpha": 1.0},
        "seed": 42,
        "out_dir": "outputs/decomp_linear_bench/test/decomp/"
    }
    
    try:
        result = run_experiment(cfg)
        print("\n✅ Decomposition test passed!")
        print(result)
        return True
    except Exception as e:
        print(f"\n❌ Decomposition test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("Quick Test for Decomposition Linear Benchmark")
    print("This will run minimal tests to verify the setup.\n")
    
    # Test baseline
    baseline_ok = test_baseline()
    
    if not baseline_ok:
        print("\n⚠️  Baseline test failed. Please fix before proceeding.")
        return False
    
    # Test decomposition
    decomp_ok = test_decomposition()
    
    if not decomp_ok:
        print("\n⚠️  Decomposition test failed. Please check tsdecomp installation.")
        return False
    
    print("\n" + "="*80)
    print("✅ All tests passed! You can now run the full experiments.")
    print("="*80)
    print("\nTo run full experiments:")
    print("  Linux/Mac: bash scripts/run_all_ett_exchange_experiments.sh")
    print("  Windows:   powershell -ExecutionPolicy Bypass -File scripts/run_all_ett_exchange_experiments.ps1")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
