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


def _extend_component(comp: np.ndarray, total_len: int, period_hint: int = None, kind: str = "trend") -> np.ndarray:
    """Extend component array to total_len by simple extrapolation/tiling.

    - trend: hold last value
    - seasonal: tile last period (period_hint) or a minimal repeating segment
    """
    comp = comp.astype(np.float32).squeeze()
    if comp.ndim != 1:
        comp = comp.flatten()
    if comp.size >= total_len:
        return comp[:total_len].copy()
    if kind.lower().startswith("trend"):
        pad_val = comp[-1]
        pad = np.full((total_len - comp.size,), pad_val, dtype=np.float32)
        return np.concatenate([comp, pad]).astype(np.float32)
    # seasonal
    per = int(period_hint) if period_hint else max(1, comp.size // 4)
    tail = comp[-per:] if per <= comp.size else comp
    reps = (total_len - comp.size + tail.size - 1) // tail.size
    ext = np.tile(tail, reps)
    return np.concatenate([comp, ext])[:total_len].astype(np.float32)

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
    decomp_mode = cfg.get("decomp_mode", "global").lower()  # "global" enforced per requirements
    
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
    
    # Run decomposition experiments (GLOBAL PATTERN ONLY)
    for decomp_cfg in decomp_methods:
        method_name = decomp_cfg.get("method", "UNKNOWN")
        print(f"\n--- Method: {method_name} (GLOBAL) ---")

        # Support multi-scale: params list
        multi_scales = decomp_cfg.get("multi_scales", None)
        param_variants = multi_scales if multi_scales else [decomp_cfg.get("params", {})]

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

                for c in range(ch):
                    # Raw sliding windows
                    X_tr_raw, Y_tr = sliding_xy_univariate(X_train[c], lookback, horizon)
                    X_te_raw, Y_te = sliding_xy_univariate(X_test[c], lookback, horizon)
                    if X_tr_raw is None or X_te_raw is None:
                        continue

                    train_len = X_train[c].shape[0]
                    val_len = X_val[c].shape[0]
                    test_len = X_test[c].shape[0]
                    total_len = train_len + val_len + test_len

                    trend_tr_wins: List[np.ndarray] = []
                    trend_te_wins: List[np.ndarray] = []
                    season_tr_wins: List[np.ndarray] = []
                    season_te_wins: List[np.ndarray] = []

                    # Precompute global components per scale
                    for param_variant in param_variants:
                        d_config = DecompositionConfig(method=method_name, params=param_variant)
                        try:
                            res = decompose(X_train[c], d_config)  # use TRAIN ONLY
                        except Exception:
                            # Fallback: use raw (no decomposition)
                            res = None

                        if res is None:
                            # pad with zeros to maintain alignment
                            trend_full = np.zeros(total_len, dtype=np.float32)
                            season_full = np.zeros(total_len, dtype=np.float32)
                        else:
                            period_hint = param_variant.get("period") or param_variant.get("block_size") or param_variant.get("window")
                            trend_full = _extend_component(res.trend, total_len, period_hint=period_hint, kind="trend")
                            season_full = _extend_component(res.season, total_len, period_hint=period_hint, kind="seasonal")

                        # Slice train/test components
                        trend_tr_full = trend_full[:train_len]
                        trend_te_full = trend_full[train_len + val_len:]
                        season_tr_full = season_full[:train_len]
                        season_te_full = season_full[train_len + val_len:]

                        X_tr_trend, _ = sliding_xy_univariate(trend_tr_full, lookback, horizon)
                        X_te_trend, _ = sliding_xy_univariate(trend_te_full, lookback, horizon)
                        X_tr_season, _ = sliding_xy_univariate(season_tr_full, lookback, horizon)
                        X_te_season, _ = sliding_xy_univariate(season_te_full, lookback, horizon)

                        if X_tr_trend is None or X_te_trend is None:
                            continue

                        trend_tr_wins.append(X_tr_trend)
                        trend_te_wins.append(X_te_trend)
                        season_tr_wins.append(X_tr_season)
                        season_te_wins.append(X_te_season)

                    # Ensure counts match raw windows
                    if not trend_tr_wins and not season_tr_wins:
                        continue
                    num_tr = X_tr_raw.shape[0]
                    num_te = X_te_raw.shape[0]
                    if (trend_tr_wins and trend_tr_wins[0].shape[0] != num_tr) or (season_tr_wins and season_tr_wins[0].shape[0] != num_tr):
                        continue
                    if (trend_te_wins and trend_te_wins[0].shape[0] != num_te) or (season_te_wins and season_te_wins[0].shape[0] != num_te):
                        continue

                    def build_feat(raw_win: np.ndarray, trend_list: List[np.ndarray], season_list: List[np.ndarray], idx: int) -> np.ndarray:
                        t_parts = [t[idx] for t in trend_list] if trend_list else []
                        s_parts = [s[idx] for s in season_list] if season_list else []

                        if ablation_mode == "+T":
                            return np.concatenate([raw_win, t_parts[0]]) if t_parts else raw_win
                        if ablation_mode == "+S":
                            return np.concatenate([raw_win, s_parts[0]]) if s_parts else raw_win
                        if ablation_mode == "+TS":
                            return np.concatenate([raw_win, t_parts[0], s_parts[0]]) if (t_parts and s_parts) else raw_win
                        if ablation_mode == "T":
                            return t_parts[0] if t_parts else raw_win
                        if ablation_mode == "S":
                            return s_parts[0] if s_parts else raw_win
                        if ablation_mode == "MS_T":
                            return np.concatenate([raw_win] + t_parts) if t_parts else raw_win
                        if ablation_mode == "MS_S":
                            return np.concatenate([raw_win] + s_parts) if s_parts else raw_win
                        if ablation_mode == "MS_TS":
                            return np.concatenate([raw_win] + t_parts + s_parts) if (t_parts or s_parts) else raw_win
                        return raw_win

                    X_tr_feats_list = [build_feat(X_tr_raw[i], trend_tr_wins, season_tr_wins, i) for i in range(num_tr)]
                    X_te_feats_list = [build_feat(X_te_raw[i], trend_te_wins, season_te_wins, i) for i in range(num_te)]

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
