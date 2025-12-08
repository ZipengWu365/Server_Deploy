import numpy as np
from typing import Dict, Any, List
from tsdecomp import decompose, DecompositionConfig

def extract_features(x_np: np.ndarray, cfg: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """
    Extract decomposition features using tsdecomp.
    
    Args:
        x_np: Input time series (1D numpy array)
        cfg: Configuration dictionary containing:
             - method: "STL" or "SSA"
             - params: dict of method parameters
             - block_sizes: list of scales (periods/windows)
    
    Returns:
        Dictionary of feature tensors.
    """
    method = cfg.get("method", "STL")
    block_sizes = cfg.get("block_sizes", [24])
    
    features = {}
    features["RAW"] = x_np.astype(np.float32)
    
    for block in block_sizes:
        # Prepare config for this scale
        decomp_params = cfg.get("params", {}).copy()
        
        if method == "STL":
            decomp_params["period"] = block
            d_cfg = DecompositionConfig(method="STL", params=decomp_params)
            res = decompose(x_np, d_cfg)
            
            features[f"T_L{block}"] = res.trend.astype(np.float32)
            features[f"S_L{block}"] = res.seasonal.astype(np.float32)
            features[f"D_L{block}"] = res.residual.astype(np.float32)
            
        elif method == "SSA":
            decomp_params["window"] = block
            d_cfg = DecompositionConfig(method="SSA", params=decomp_params)
            res = decompose(x_np, d_cfg)
            
            features[f"SSA_T_L{block}"] = res.trend.astype(np.float32)
            features[f"SSA_S_L{block}"] = res.seasonal.astype(np.float32)
            # SSA residual might be noise, or we might want specific components
            # For now, map residual to D (Dispersion/Noise)
            features[f"SSA_D_L{block}"] = res.residual.astype(np.float32)
            
    return features
