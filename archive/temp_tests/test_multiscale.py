"""Quick test for multi-scale feature extraction"""
import numpy as np
import sys
sys.path.insert(0, r'c:\Users\wzp07\Downloads\LongTermTS\Time-Series-Library-main\Server_Deploy-main')

from features.decomp_linear_bench.builder import build_features

# Generate test data
np.random.seed(42)
x_window = np.sin(np.linspace(0, 10*np.pi, 336)) + np.random.randn(336) * 0.1

# Test multi-scale configuration
cfg = {
    "tsdecomp": {
        "method": "STL",
        "scales": [
            {"period": 24},
            {"period": 168}
        ]
    }
}
"""Quick test for multi-scale feature extraction"""
import numpy as np
import sys
sys.path.insert(0, r'c:\Users\wzp07\Downloads\LongTermTS\Time-Series-Library-main\Server_Deploy-main')

from features.decomp_linear_bench.builder import build_features

# Generate test data
np.random.seed(42)
x_window = np.sin(np.linspace(0, 10*np.pi, 336)) + np.random.randn(336) * 0.1

# Test multi-scale configuration
cfg = {
    "tsdecomp": {
        "method": "STL",
        "scales": [
            {"period": 24},
            {"period": 168}
        ]
    }
}

print("Testing multi-scale feature extraction...")
print(f"Input window shape: {x_window.shape}")

try:
    result = build_features(x_window, cfg)
    print(f"\n[SUCCESS]")
    print(f"Output feature shape: {result['X'].shape}")
    print(f"Number of components: {len(result['meta']['component_names'])}")
    print(f"\nFirst 10 component names:")
    for name in result['meta']['component_names'][:10]:
        print(f"  - {name}")
    print(f"\nLast 10 component names:")
    for name in result['meta']['component_names'][-10:]:
        print(f"  - {name}")
except Exception as e:
    print(f"\n[ERROR]: {e}")
    import traceback
    traceback.print_exc()
