import numpy as np
from typing import Dict, Any, List
from tsdecomp import decompose, DecompositionConfig

def build_features(x_window: np.ndarray, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build multi-scale decomposition features for a single window.
    
    Supports two configuration formats:
    1. Legacy format: {"tsdecomp": {"method": "STL", "params": {"period": 24}}}
    2. Multi-scale format: {"tsdecomp": {"method": "STL", "scales": [{"period": 24}, {"period": 168}, ...]}}
    3. Multiple methods: {"tsdecomp": [{"method": "STL", "scales": [...]}, {"method": "SSA", "scales": [...]}]}
    
    Args:
        x_window: Input time series window (1D array).
        cfg: Configuration dict
             
    Returns:
        Dict with keys:
        - "X": Feature vector (1D array)
        - "meta": Metadata about components
    """
    tsdecomp_cfg = cfg.get("tsdecomp", {})
    
    # Normalize to list of method configs
    if isinstance(tsdecomp_cfg, list):
        method_configs = tsdecomp_cfg
    elif isinstance(tsdecomp_cfg, dict) and tsdecomp_cfg:
        method_configs = [tsdecomp_cfg]
    else:
        method_configs = []
    
    all_feats = []
    component_names = []
    
    # Always include RAW as baseline
    all_feats.append(x_window)
    component_names.extend([f"RAW_{i}" for i in range(len(x_window))])
    
    # If no decomposition, return RAW only
    if not method_configs:
        X_feat = np.concatenate(all_feats)
        return {
            "X": X_feat.astype(np.float32),
            "meta": {"component_names": component_names}
        }
    
    # Process each method configuration
    for method_cfg in method_configs:
        method_name = method_cfg.get("method", "STL")
        
        # Check if multi-scale format (has "scales" key)
        if "scales" in method_cfg:
            scales = method_cfg["scales"]
            if not isinstance(scales, list):
                scales = [scales]
        # Legacy single-scale format (has "params" key)
        elif "params" in method_cfg:
            scales = [method_cfg["params"]]
        else:
            print(f"Warning: No scales or params found for {method_name}, skipping")
            continue
        
        # Decompose at each scale
        for scale_idx, scale_params in enumerate(scales):
            # Infer scale identifier for naming
            scale_id = _infer_scale_id(scale_params)
            
            # Create decomposition config
            d_config = DecompositionConfig(method=method_name, params=scale_params)
            
            try:
                res = decompose(x_window, d_config)
            except Exception as e:
                print(f"Warning: Decomposition failed for {method_name} scale {scale_id}: {e}")
                continue
            
            if res:
                # Naming convention: {Method}_{Component}_L{Scale}_{idx}
                # Trend
                all_feats.append(res.trend)
                component_names.extend([f"{method_name}_T_L{scale_id}_{i}" for i in range(len(res.trend))])
                
                # Seasonal
                all_feats.append(res.season)
                component_names.extend([f"{method_name}_S_L{scale_id}_{i}" for i in range(len(res.season))])
                
                # Residual (Dispersion)
                all_feats.append(res.residual)
                component_names.extend([f"{method_name}_D_L{scale_id}_{i}" for i in range(len(res.residual))])
                
                # Per-scale statistics
                if len(res.trend) > 1:
                    slope = (res.trend[-1] - res.trend[0]) / len(res.trend)
                    all_feats.append(np.array([slope]))
                    component_names.append(f"{method_name}_TrendSlope_L{scale_id}")
                
                disp_std = np.std(res.residual)
                all_feats.append(np.array([disp_std]))
                component_names.append(f"{method_name}_DispStd_L{scale_id}")
                
                season_energy = np.std(res.season)
                all_feats.append(np.array([season_energy]))
                component_names.append(f"{method_name}_SeasonEnergy_L{scale_id}")

    if len(all_feats) == 1:  # Only RAW
        print("Warning: No decomposition components extracted, using RAW only")

    X_feat = np.concatenate(all_feats)
    
    return {
        "X": X_feat.astype(np.float32),
        "meta": {
            "component_names": component_names,
        }
    }

def _infer_scale_id(params: Dict[str, Any]) -> str:
    """
    Infer a scale identifier from decomposition parameters.
    
    Returns a string like "24" for period=24, "168" for window=168, etc.
    """
    # Common scale parameters across methods
    for key in ["period", "window", "block_size", "level"]:
        if key in params:
            return str(params[key])
    
    # Fallback: hash of params
    return str(hash(frozenset(params.items())))[:4]

def build_dataset(series: np.ndarray, lookback: int, horizon: int, cfg: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """
    Create (X, y) dataset by sliding window.
    """
    X_list = []
    y_list = []
    meta_list = []
    
    n_samples = len(series) - lookback - horizon + 1
    step = 1 
    
    for i in range(0, n_samples, step):
        window = series[i : i + lookback]
        target = series[i + lookback + horizon - 1]
        
        feat_res = build_features(window, cfg)
        X_list.append(feat_res["X"])
        y_list.append(target)
        if i == 0:
            meta_list = feat_res["meta"]
            
    X = np.stack(X_list)
    y = np.array(y_list)
    
    return {"X": X, "y": y, "meta": meta_list}
