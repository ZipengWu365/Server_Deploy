"""
Decomposition Linear Benchmark Runner
Channel-wise modeling with proper data protocol.
"""
import os
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from typing import Dict, Any, List, Tuple

# Standard splits
STANDARD_SPLITS = {
    'etth1': (8545, 2881, 2881),
    'etth2': (8545, 2881, 2881),
    'ettm1': (34465, 11521, 11521),
    'ettm2': (34465, 11521, 11521),
    'electricity': (18317, 2633, 5261),
    'traffic': (12185, 1757, 3509),
    'weather': (36792, 5271, 10540),
    'exchange_rate': (5120, 665, 1422),
}

def load_csv(path: str) -> Tuple[np.ndarray, List[str]]:
    """Load CSV - shape (channels, T)"""
    df = pd.read_csv(path, header=0, index_col=0)
    df = df.apply(pd.to_numeric, errors='coerce').ffill().fillna(0.0)
    arr = df.values.T.astype(np.float32)  # (channels, T)
    return arr, list(df.columns)

def zscore_channelwise(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Z-score normalization per channel on ENTIRE series"""
    mu = X.mean(axis=1, keepdims=True, dtype=np.float32)
    std = X.std(axis=1, keepdims=True, dtype=np.float32) + 1e-8
    Xn = (X - mu) / std
    return Xn.astype(np.float32), mu.squeeze(), std.squeeze()

def split_by_standard(X: np.ndarray, dataset_name: str, split_config: List = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    """Split data using standard splits or config-specified split"""
    ch, T = X.shape
    
    # Use config split if provided
    if split_config and len(split_config) == 3:
        tr, va, te = split_config
        total = tr + va + te
        if total <= T:
            X_tr = X[:, :tr].astype(np.float32)
            X_va = X[:, tr:tr+va].astype(np.float32)
            X_te = X[:, tr+va:tr+va+te].astype(np.float32)
            return X_tr, X_va, X_te, True
    
    # Fallback to standard splits
    name = os.path.basename(dataset_name).lower()
    key = None
    for k in STANDARD_SPLITS.keys():
        if k in name:
            key = k
            break
    
    if key is None:
        t1 = int(T * 0.7)
        t2 = int(T * 0.85)
        return X[:, :t1], X[:, t1:t2], X[:, t2:], False
    
    tr, va, te = STANDARD_SPLITS[key]
    total = tr + va + te
    
    if total > T:
        t1 = int(T * 0.7)
        t2 = int(T * 0.85)
        return X[:, :t1], X[:, t1:t2], X[:, t2:], False
    
    X_tr = X[:, :tr].astype(np.float32)
    X_va = X[:, tr:tr+va].astype(np.float32)
    X_te = X[:, tr+va:tr+va+te].astype(np.float32)
    
    return X_tr, X_va, X_te, True

def sliding_xy_univariate(series: np.ndarray, input_len: int, out_len: int) -> Tuple[np.ndarray, np.ndarray]:
    """Create supervised data using sliding window"""
    series = series.astype(np.float32)
    total = input_len + out_len
    
    if series.shape[0] < total:
        return None, None
    
    win = sliding_window_view(series, total)
    X = win[:, :input_len]
    Y = win[:, input_len:]
    
    return X, Y

def run_experiment(cfg: Dict[str, Any]):
    """
    Run experiment with decomposition methods and ablation modes.
    Channel-wise modeling with proper data protocol.
    """
    from .builder import build_features
    from .ablations import make_ablation_mask
    from tsdecomp import decompose, DecompositionConfig
    
    out_dir = cfg.get("out_dir", "outputs/decomp_linear_bench/")
    os.makedirs(out_dir, exist_ok=True)
    
    print("="*80)
    print(f"Experiment: {cfg.get('dataset', {}).get('name', 'Unknown')}")
    print(f"Output: {out_dir}")
    print("="*80)
    
    dataset_cfg = cfg["dataset"]
    dataset_name = dataset_cfg["name"]
    dataset_path = dataset_cfg["path"]
    
    # Load data - shape (channels, T)
    Xraw, cols = load_csv(dataset_path)
    ch, T = Xraw.shape
    
    print(f"Data: {ch} channels × {T} timesteps")
    print(f"Columns: {cols}")
    
    # Z-score normalize ENTIRE series per channel
    Xs, mu, std = zscore_channelwise(Xraw)
    
    # Split using standard splits or config split
    split_config = dataset_cfg.get("split", None)
    X_train, X_val, X_test, used_standard = split_by_standard(Xs, dataset_path, split_config)
    
    print(f"Split: {'standard' if used_standard else 'ratio'}")
    print(f"Train: {X_train.shape[1]}, Val: {X_val.shape[1]}, Test: {X_test.shape[1]}")
    
    lookback = cfg["lookback"]
    horizons = cfg["horizons"]
    learner_params = cfg.get("learner_params", {})
    alpha = learner_params.get("alpha", 1.0)
    
    # Get decomposition and ablation configurations
    decomp_methods = cfg.get("decomp", [])
    ablation_modes = cfg.get("ablation_modes", ["RAW"])
    
    all_results = []
    
    # Always run baseline first
    print("\n--- BASELINE (RAW) ---")
    for horizon in horizons:
        sse_total = 0.0
        sae_total = 0.0
        mape_sum = 0.0
        n_total = 0
        
        # Process each channel independently
        for c in range(ch):
            # Create sliding windows
            X_tr, Y_tr = sliding_xy_univariate(X_train[c], lookback, horizon)
            X_te, Y_te = sliding_xy_univariate(X_test[c], lookback, horizon)
            
            if X_tr is None or X_te is None:
                continue
            
            # Standardize features
            scaler = StandardScaler(with_mean=True, with_std=True)
            X_tr_scaled = scaler.fit_transform(X_tr.astype(np.float32))
            X_te_scaled = scaler.transform(X_te.astype(np.float32))
            
            # Train Ridge regression
            model = Ridge(alpha=alpha, solver='auto')
            model.fit(X_tr_scaled, Y_tr.astype(np.float32))
            
            # Predict
            Y_pred = model.predict(X_te_scaled).astype(np.float32)
            
            # Calculate errors
            diff = Y_pred - Y_te.astype(np.float32)
            sse_total += float(np.sum(diff * diff))
            sae_total += float(np.sum(np.abs(diff)))
            mape_sum += float(np.sum(np.abs(diff) / (np.abs(Y_te) + 1e-8)))
            n_total += int(diff.size)
        
        if n_total == 0:
            mse = np.nan
            mae = np.nan
            mape = np.nan
        else:
            mse = sse_total / n_total
            mae = sae_total / n_total
            mape = 100.0 * mape_sum / n_total
        
        print(f"  [BASELINE] L={lookback:3d} H={horizon:3d} | MSE={mse:.4f}  MAE={mae:.4f}  MAPE={mape:.2f}%")
        
        all_results.append({
            "dataset": dataset_name, "decomp": "NONE", "horizon": horizon,
            "ablation": "RAW", "learner": "Ridge", "n_feat": lookback,
            "MAE": mae, "MSE": mse, "MAPE": mape
        })
    
    # If no decomposition methods specified, return baseline only
    if not decomp_methods or ablation_modes == ["RAW"]:
        res_df = pd.DataFrame(all_results)
        out_csv = os.path.join(out_dir, "metrics_summary_by_method_horizon_ablation.csv")
        res_df.to_csv(out_csv, index=False)
        print(f"\nSaved: {out_csv}")
        return res_df
    
    # Run decomposition experiments
    for decomp_cfg in decomp_methods:
        method_name = decomp_cfg.get("method", "UNKNOWN")
        print(f"\n--- Method: {method_name} ---")
        
        for ablation_mode in ablation_modes:
            if ablation_mode == "RAW":
                continue  # Already done in baseline
            
            print(f"  Ablation: {ablation_mode}")
            
            for horizon in horizons:
                sse_total = 0.0
                sae_total = 0.0
                mape_sum = 0.0
                n_total = 0
                n_feat_total = 0
                
                # Process each channel independently
                for c in range(ch):
                    # Create sliding windows for raw data
                    X_tr_raw, Y_tr = sliding_xy_univariate(X_train[c], lookback, horizon)
                    X_te_raw, Y_te = sliding_xy_univariate(X_test[c], lookback, horizon)
                    
                    if X_tr_raw is None or X_te_raw is None:
                        continue
                    
                    # Decompose each window independently (NO DATA LEAKAGE)
                    X_tr_feats_list = []
                    X_te_feats_list = []
                    
                    d_config = DecompositionConfig(method=method_name, params=decomp_cfg.get("params", {}))
                    
                    # Process training windows
                    for i in range(X_tr_raw.shape[0]):
                        window = X_tr_raw[i]
                        
                        try:
                            res = decompose(window, d_config)
                            
                            # Build feature vector based on ablation mode
                            if ablation_mode == "+T":
                                feat = np.concatenate([window, res.trend])
                            elif ablation_mode == "+S":
                                feat = np.concatenate([window, res.season])
                            elif ablation_mode == "T":
                                feat = res.trend
                            elif ablation_mode == "S":
                                feat = res.season
                            else:
                                feat = window
                            
                            X_tr_feats_list.append(feat)
                        except Exception as e:
                            # If decomposition fails, use raw window
                            X_tr_feats_list.append(window)
                    
                    # Process test windows
                    for i in range(X_te_raw.shape[0]):
                        window = X_te_raw[i]
                        
                        try:
                            res = decompose(window, d_config)
                            
                            # Build feature vector based on ablation mode
                            if ablation_mode == "+T":
                                feat = np.concatenate([window, res.trend])
                            elif ablation_mode == "+S":
                                feat = np.concatenate([window, res.season])
                            elif ablation_mode == "T":
                                feat = res.trend
                            elif ablation_mode == "S":
                                feat = res.season
                            else:
                                feat = window
                            
                            X_te_feats_list.append(feat)
                        except Exception as e:
                            # If decomposition fails, use raw window
                            X_te_feats_list.append(window)
                    
                    if not X_tr_feats_list or not X_te_feats_list:
                        continue
                    
                    X_tr_feats = np.array(X_tr_feats_list, dtype=np.float32)
                    X_te_feats = np.array(X_te_feats_list, dtype=np.float32)
                    
                    n_feat_total = X_tr_feats.shape[1]
                    
                    # Standardize features
                    scaler = StandardScaler(with_mean=True, with_std=True)
                    X_tr_scaled = scaler.fit_transform(X_tr_feats.astype(np.float32))
                    X_te_scaled = scaler.transform(X_te_feats.astype(np.float32))
                    
                    # Train Ridge regression
                    model = Ridge(alpha=alpha, solver='auto')
                    model.fit(X_tr_scaled, Y_tr.astype(np.float32))
                    
                    # Predict
                    Y_pred = model.predict(X_te_scaled).astype(np.float32)
                    
                    # Calculate errors
                    diff = Y_pred - Y_te.astype(np.float32)
                    sse_total += float(np.sum(diff * diff))
                    sae_total += float(np.sum(np.abs(diff)))
                    mape_sum += float(np.sum(np.abs(diff) / (np.abs(Y_te) + 1e-8)))
                    n_total += int(diff.size)
                
                if n_total == 0:
                    mse = np.nan
                    mae = np.nan
                    mape = np.nan
                else:
                    mse = sse_total / n_total
                    mae = sae_total / n_total
                    mape = 100.0 * mape_sum / n_total
                
                print(f"    [{method_name} {ablation_mode}] H={horizon:3d} | MSE={mse:.4f}  MAE={mae:.4f}  MAPE={mape:.2f}%")
                
                all_results.append({
                    "dataset": dataset_name, "decomp": method_name, "horizon": horizon,
                    "ablation": ablation_mode, "learner": "Ridge", "n_feat": n_feat_total,
                    "MAE": mae, "MSE": mse, "MAPE": mape
                })
    
    # Save
    res_df = pd.DataFrame(all_results)
    out_csv = os.path.join(out_dir, "metrics_summary_by_method_horizon_ablation.csv")
    res_df.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")
    
    return res_df
