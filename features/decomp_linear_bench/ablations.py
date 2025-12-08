import numpy as np
from typing import List, Dict, Any

def make_ablation_mask(meta: Dict[str, Any], mode: str) -> np.ndarray:
    """
    Create a boolean mask for feature selection based on ablation mode.
    
    Supports:
    - Component ablation: "T", "S", "D", "T+S", "T+D", "S+D", "T+S+D"
    - Concatenation modes: "+T" (RAW+Trend), "+S" (RAW+Season)
    - Scale-specific: "T_L24", "S_L168", "D_L720"
    - Cross-scale: "T_L24+S_L168"
    - RAW only: "RAW"
    
    Args:
        meta: Metadata from builder (contains "component_names").
        mode: Ablation mode string.
        
    Returns:
        Boolean mask array of shape (n_features,).
    """
    names = meta.get("component_names", [])
    n_feat = len(names)
    mask = np.zeros(n_feat, dtype=bool)
    
    # Handle concatenation modes: +T, +S, +R (RAW + component)
    if mode.startswith("+"):
        # Always include RAW
        for i, name in enumerate(names):
            if "RAW" in name:
                mask[i] = True
        
        # Map +T/+S/+R to component types
        comp_map = {
            "+T": "T",  # Trend
            "+S": "S",  # Season
            "+R": "D",  # Residual (D for dispersion)
        }
        
        comp_type = comp_map.get(mode)
        if comp_type:
            for i, name in enumerate(names):
                if f"_{comp_type}_L" in name:
                    mask[i] = True
                # Also match statistics
                if comp_type == "T" and ("TrendSlope" in name):
                    mask[i] = True
                if comp_type == "S" and ("SeasonEnergy" in name):
                    mask[i] = True
                if comp_type == "D" and ("DispStd" in name):
                    mask[i] = True
        return mask
    
    # Parse mode: split by + for multiple selections
    selectors = mode.split("+")
    
    for selector in selectors:
        selector = selector.strip()
        
        # Check if it's a scale-specific selector (e.g., "T_L24")
        if "_L" in selector:
            # Extract component type and scale
            parts = selector.split("_L")
            if len(parts) == 2:
                comp_type = parts[0]
                scale = parts[1]
                
                # Match names containing both component type and scale
                for i, name in enumerate(names):
                    if f"_{comp_type}_L{scale}_" in name:
                        mask[i] = True
                    elif f"Slope_L{scale}" in name and comp_type == "T":
                        mask[i] = True
                    elif f"DispStd_L{scale}" in name and comp_type == "D":
                        mask[i] = True
                    elif f"SeasonEnergy_L{scale}" in name and comp_type == "S":
                        mask[i] = True
        # Standard component-level selector
        else:
            for i, name in enumerate(names):
                is_trend = "Trend" in name or "_T_L" in name or "TrendSlope" in name
                is_season = "Season" in name or "_S_L" in name or "SeasonEnergy" in name
                is_disp = "Residual" in name or "_D_L" in name or "Disp" in name or "Noise" in name
                is_raw = "RAW" in name
                
                include = False
                if selector == "T" and is_trend: include = True
                if selector == "S" and is_season: include = True
                if selector == "D" and is_disp: include = True
                if selector == "R" and is_disp: include = True  # R is alias for D
                if selector == "RAW" and is_raw: include = True
                
                if include:
                    mask[i] = True
        
    return mask
