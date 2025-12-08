import numpy as np
from typing import Optional, List, Dict, Any
from ..core import DecompositionConfig, DecompResult
from ..registry import register

try:
    from PyEMD import EMD
except ImportError:
    EMD = None

def _dominant_frequency(x: np.ndarray, fs: float = 1.0) -> float:
    x = np.asarray(x, dtype=float)
    if len(x) < 2: return 0.0
    x = x - np.mean(x)
    spectrum = np.abs(np.fft.rfft(x))
    freqs = np.fft.rfftfreq(len(x), d=1.0 / fs if fs > 0 else 1.0)
    if spectrum.size <= 1: return 0.0
    idx = int(np.argmax(spectrum[1:]) + 1) if spectrum.size > 1 else 0
    return float(freqs[idx]) if idx < len(freqs) else 0.0

def _aggregate_modes(modes: np.ndarray, indices: Optional[List[int]]) -> np.ndarray:
    if indices is None or len(indices) == 0:
        return np.zeros(modes.shape[1], dtype=float)
    valid = [idx for idx in indices if 0 <= idx < modes.shape[0]]
    if not valid:
        return np.zeros(modes.shape[1], dtype=float)
    return np.sum(modes[valid, :], axis=0)

@register("EMD")
def emd_decompose(x: np.ndarray, cfg: DecompositionConfig) -> DecompResult:
    if EMD is None:
        raise ImportError("PyEMD is required for EMD decomposition.")

    y_arr = np.asarray(x, dtype=float)
    T = y_arr.shape[0]
    
    fs = float(cfg.params.get("fs", 1.0))
    primary_period = cfg.params.get("primary_period")
    primary_period = float(primary_period) if primary_period not in (None, 0) else None
    n_imfs = cfg.params.get("n_imfs")

    emd = EMD()
    if n_imfs is not None:
        imfs = emd.emd(y_arr, max_imf=int(n_imfs))
    else:
        imfs = emd.emd(y_arr)
        
    imfs = np.asarray(imfs, dtype=float)
    if imfs.ndim == 1: imfs = imfs[np.newaxis, :]
    num_imfs = imfs.shape[0]
    
    if num_imfs == 0:
        zeros = np.zeros_like(y_arr)
        return DecompResult(zeros, zeros, y_arr.copy(), extra={"imfs": []})

    dom_freqs = [_dominant_frequency(comp, fs=fs) for comp in imfs]
    trend_imfs = list(cfg.params.get("trend_imfs", []))
    season_imfs = list(cfg.params.get("season_imfs", []))

    if not trend_imfs and not season_imfs:
        if primary_period is not None and primary_period > 0:
            f0 = 1.0 / primary_period
            tol = float(cfg.params.get("season_freq_tol_ratio", 0.25)) * f0
            low_thresh = float(cfg.params.get("trend_freq_threshold", f0 / 4.0 if f0 else 0.05))

            for idx, f_dom in enumerate(dom_freqs):
                if f_dom <= max(low_thresh, 1e-8):
                    trend_imfs.append(idx)
                elif f0 > 0 and abs(f_dom - f0) <= max(tol, 1e-8):
                    season_imfs.append(idx)

            if not trend_imfs: trend_imfs.append(num_imfs - 1)
            if not season_imfs:
                best_idx = int(np.argmin([abs(f - f0) for f in dom_freqs]))
                season_imfs.append(best_idx)
        else:
            if num_imfs >= 1: trend_imfs.append(num_imfs - 1)
            if num_imfs >= 2: trend_imfs.append(num_imfs - 2)
            if num_imfs >= 1: season_imfs.append(0)
            if num_imfs >= 3: season_imfs.append(1)

    trend = _aggregate_modes(imfs, trend_imfs)
    season = _aggregate_modes(imfs, season_imfs)
    residual = y_arr - trend - season

    return DecompResult(
        trend=trend,
        season=season,
        residual=residual,
        meta={"imfs": imfs, "trend_imfs": trend_imfs, "season_imfs": season_imfs}
    )
