import numpy as np
from typing import List, Optional, Dict, Any
from ..core import DecompositionConfig, DecompResult
from ..registry import register

def _basic_ssa(y: np.ndarray, window: int, rank: int) -> List[np.ndarray]:
    """
    Basic SSA: build Hankel matrix, run SVD, reconstruct RCs via diagonal averaging.
    """
    y_arr = np.asarray(y, dtype=float)
    T = y_arr.shape[0]
    L = int(window)
    if L < 2 or L > T - 1:
        # Fallback or error? Original code raises ValueError
        # We'll clamp it to be safe if possible, or raise
        if T < 3: return [] # Too short
        L = min(max(2, L), T - 1)
        
    K = T - L + 1

    X = np.empty((L, K), dtype=float)
    for i in range(K):
        X[:, i] = y_arr[i : i + L]

    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    d = min(rank, U.shape[1])
    rc_list: List[np.ndarray] = []
    for idx in range(d):
        Xi = np.outer(U[:, idx], s[idx] * Vt[idx, :])
        rc = _diagonal_averaging(Xi)[:T]
        rc_list.append(rc)
    return rc_list

def _diagonal_averaging(matrix: np.ndarray) -> np.ndarray:
    L, K = matrix.shape
    T = L + K - 1
    recon = np.zeros(T)
    counts = np.zeros(T)
    for i in range(L):
        for j in range(K):
            recon[i + j] += matrix[i, j]
            counts[i + j] += 1.0
    counts[counts == 0.0] = 1.0
    return recon / counts

def _sum_components(components: List[np.ndarray], indices: List[int], length: int) -> np.ndarray:
    if not indices:
        return np.zeros(length)
    out = np.zeros(length)
    for idx in indices:
        if 0 <= idx < len(components):
            out += components[idx]
    return out

def _dominant_frequency(x: np.ndarray, fs: float = 1.0) -> float:
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return 0.0
    x = x - np.mean(x)
    spectrum = np.abs(np.fft.rfft(x))
    freqs = np.fft.rfftfreq(len(x), d=1.0 / fs if fs > 0 else 1.0)
    if spectrum.size <= 1:
        return 0.0
    idx = int(np.argmax(spectrum[1:]) + 1) if spectrum.size > 1 else 0
    return float(freqs[idx]) if idx < len(freqs) else 0.0

@register("SSA")
def ssa_decompose(x: np.ndarray, cfg: DecompositionConfig) -> DecompResult:
    """
    SSA-based decomposition.
    """
    y_arr = np.asarray(x, dtype=float)
    T = y_arr.shape[0]
    
    # Params
    window = int(cfg.params.get("window", max(4, T // 4)))
    window = min(max(2, window), T - 1)
    rank = int(cfg.params.get("rank", 10))
    fs = float(cfg.params.get("fs", 1.0))
    primary_period = cfg.params.get("primary_period")
    primary_period = float(primary_period) if primary_period not in (None, 0) else None

    rc_list = _basic_ssa(y_arr, window=window, rank=rank)
    num_rc = len(rc_list)
    
    if num_rc == 0:
        zeros = np.zeros_like(y_arr)
        return DecompResult(zeros, zeros, y_arr.copy(), extra={"rc_list": []})

    trend_components = list(cfg.params.get("trend_components", []))
    season_components = list(cfg.params.get("season_components", []))
    
    # Auto-grouping logic from decomp_methods.py
    if not trend_components and not season_components:
        if primary_period is not None and primary_period > 0:
            dom_freqs = [_dominant_frequency(rc, fs=fs) for rc in rc_list]
            f0 = 1.0 / primary_period
            tol = float(cfg.params.get("season_freq_tol_ratio", 0.25)) * f0
            low_thresh = float(cfg.params.get("trend_freq_threshold", f0 / 4.0 if f0 else 0.05))
            
            for idx, f_dom in enumerate(dom_freqs):
                if f_dom <= max(low_thresh, 1e-8):
                    trend_components.append(idx)
                elif abs(f_dom - f0) <= max(tol, 1e-8):
                    season_components.append(idx)
            
            # Fallbacks
            if not trend_components and num_rc >= 1: trend_components.append(0)
            if not season_components:
                for idx in range(num_rc):
                    if idx not in trend_components:
                        season_components.append(idx)
                        break
        else:
            # Simple heuristic
            if num_rc >= 1: trend_components.append(0)
            if num_rc >= 2: trend_components.append(1)
            if num_rc >= 4: season_components.extend([2, 3])
            elif num_rc >= 3: season_components.append(2)

    trend = _sum_components(rc_list, trend_components, T)
    season = _sum_components(rc_list, season_components, T)
    residual = y_arr - trend - season

    return DecompResult(
        trend=trend,
        season=season,
        residual=residual,
        meta={
            "rc_list": rc_list,
            "trend_components": trend_components,
            "season_components": season_components
        }
    )
