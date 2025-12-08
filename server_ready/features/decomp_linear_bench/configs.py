# Method presets
PRESETS = {
    "STD": {
        "method": "STD",
        "params": {"block_size": 24, "seasonal_rank": 6}
    },
    "STD_MULTI": {
        "method": "STD", 
        "params": {} # Placeholder, overridden by get_decomp_config
    },
    "STD_FULL": {
        "method": "STD",
        "params": {} # Placeholder, overridden by get_decomp_config
    },
    "SSA": {
        "method": "SSA",
        "params": {"window": 336, "rank": 10}
    },
    "STL": {
        "method": "STL",
        "params": {"period": 24}
    },
    "VMD": {
        "method": "VMD", 
        "params": {}
    },
    "CEEMDAN": {
        "method": "CEEMDAN", 
        "params": {}
    },
    "WAVELET": {
        "method": "Wavelet",
        "params": {"wavelet": "db4", "level": 3}
    },
    "EMD": {
        "method": "EMD",
        "params": {}
    },
    "MA": {
        "method": "MA",
        "params": {"window": 24}
    },
    "GABOR_CLUSTER": {
        "method": "GaborCluster",
        "params": {
            "model_path": "models/gabor_cluster_v1.npz", 
            "max_clusters": 6
        }
    }
}

def get_decomp_config(preset_name: str, dataset_name: str = None):
    if preset_name == "STL_MULTI":
        # Use same scales as STD
        std_cfg = get_decomp_config("STD_MULTI", dataset_name)
        return [{"method": "STL", "params": {"period": c["params"]["block_size"]}} for c in std_cfg]

    if preset_name == "SSA_MULTI":
        std_cfg = get_decomp_config("STD_MULTI", dataset_name)
        return [{"method": "SSA", "params": {"window": c["params"]["block_size"], "rank": 10}} for c in std_cfg]

    if preset_name == "MA_MULTI":
        std_cfg = get_decomp_config("STD_MULTI", dataset_name)
        return [{"method": "MA", "params": {"window": c["params"]["block_size"]}} for c in std_cfg]

    if preset_name == "STD_FULL":
        # Fixed scales [24, 168, 1440], ranks [8, 6, 6]
        return [
            {"method": "STD", "params": {"block_size": 24, "seasonal_rank": 8}},
            {"method": "STD", "params": {"block_size": 168, "seasonal_rank": 6}},
            {"method": "STD", "params": {"block_size": 1440, "seasonal_rank": 6}},
        ]
    
    if preset_name == "STD_MULTI":
        if not dataset_name:
             return [
                {"method": "STD", "params": {"block_size": 24, "seasonal_rank": 6}},
                {"method": "STD", "params": {"block_size": 168, "seasonal_rank": 6}},
             ]
        
        ds = dataset_name.lower()
        if "etth" in ds:
             # [24, 168, 504, 1440], ranks [6, 5, 5, 5]
             return [
                {"method": "STD", "params": {"block_size": 24, "seasonal_rank": 6}},
                {"method": "STD", "params": {"block_size": 168, "seasonal_rank": 5}},
                {"method": "STD", "params": {"block_size": 504, "seasonal_rank": 5}},
                {"method": "STD", "params": {"block_size": 1440, "seasonal_rank": 5}},
             ]
        elif "ettm" in ds:
             # [96, 672, 1440], ranks [8, 6, 6]
             return [
                {"method": "STD", "params": {"block_size": 96, "seasonal_rank": 8}},
                {"method": "STD", "params": {"block_size": 672, "seasonal_rank": 6}},
                {"method": "STD", "params": {"block_size": 1440, "seasonal_rank": 6}},
             ]
        elif "exchange" in ds:
             # [7, 30, 1440], ranks [2, 3, 3]
             return [
                {"method": "STD", "params": {"block_size": 7, "seasonal_rank": 2}},
                {"method": "STD", "params": {"block_size": 30, "seasonal_rank": 3}},
                {"method": "STD", "params": {"block_size": 1440, "seasonal_rank": 3}},
             ]
        elif "electricity" in ds or "traffic" in ds or "weather" in ds:
             # [24, 168, 1440], ranks [8, 6, 6] (approx for elec)
             return [
                {"method": "STD", "params": {"block_size": 24, "seasonal_rank": 8}},
                {"method": "STD", "params": {"block_size": 168, "seasonal_rank": 6}},
                {"method": "STD", "params": {"block_size": 1440, "seasonal_rank": 6}},
             ]
        else:
             return [
                {"method": "STD", "params": {"block_size": 24, "seasonal_rank": 6}},
                {"method": "STD", "params": {"block_size": 168, "seasonal_rank": 6}},
             ]

    # For single presets, return as single-element list for consistency
    cfg = PRESETS.get(preset_name, {}).copy()
    if cfg:
        return [cfg]
    return []
