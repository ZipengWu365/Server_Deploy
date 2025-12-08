import numpy as np
from typing import Optional, List, Dict, Any
from ..core import DecompositionConfig, DecompResult
from ..registry import register

try:
    import pywt
except ImportError:
    pywt = None

def _reconstruct_from_levels(coeffs: List[np.ndarray], keep_levels: List[int], wavelet: str, target_len: int) -> np.ndarray:
    rec_coeffs: List[Optional[np.ndarray]] = []
    for idx, coeff in enumerate(coeffs):
        if idx in (keep_levels or []):
            rec_coeffs.append(np.copy(coeff))
        else:
            rec_coeffs.append(np.zeros_like(coeff))
    recon = pywt.waverec(rec_coeffs, wavelet)
    if recon.shape[0] > target_len:
        recon = recon[:target_len]
    elif recon.shape[0] < target_len:
        pad = target_len - recon.shape[0]
        recon = np.pad(recon, (0, pad), mode="edge")
    return recon

@register("Wavelet")
def wavelet_decompose(x: np.ndarray, cfg: DecompositionConfig) -> DecompResult:
    if pywt is None:
        raise ImportError("PyWavelets (pywt) is required for wavelet decomposition.")

    y = np.asarray(x, dtype=float)
    wavelet_name = cfg.params.get("wavelet", "db4")
    level = cfg.params.get("level")
    
    wavelet = pywt.Wavelet(wavelet_name)
    max_level = pywt.dwt_max_level(len(y), wavelet.dec_len)
    if level is None:
        level = max(1, min(5, max_level))
        
    coeffs = pywt.wavedec(y, wavelet, level=level)
    num_coeffs = len(coeffs)

    trend_levels = cfg.params.get("trend_levels")
    season_levels = cfg.params.get("season_levels")

    if trend_levels is None:
        trend_levels = [0]
    if season_levels is None and num_coeffs > 2:
        season_levels = [1, 2]
    elif season_levels is None:
        season_levels = [idx for idx in range(1, num_coeffs)]

    trend = _reconstruct_from_levels(coeffs, trend_levels, wavelet_name, len(y))
    season = _reconstruct_from_levels(coeffs, season_levels, wavelet_name, len(y))
    residual = y - trend - season

    return DecompResult(
        trend=trend,
        season=season,
        residual=residual,
        meta={"coeffs": coeffs, "params": cfg.params}
    )
