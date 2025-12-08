from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from ..core import DecompositionConfig, DecompResult
from ..registry import register

@dataclass
class GaborConfig:
    fs: float = 1.0
    win_len: int = 256
    hop: int = 64
    n_fft: Optional[int] = None
    window_type: str = "gaussian"
    gaussian_sigma: Optional[float] = None
    bands: Optional[List[Tuple[float,float]]] = None
    ridge: bool = False
    ridge_max_peaks: int = 2
    tight_frame: bool = True

def _make_window(L:int, wtype:str, sigma:Optional[float])->np.ndarray:
    if wtype == "gaussian":
        if sigma is None:
            sigma = L/6.0
        n = np.arange(L) - (L-1)/2
        w = np.exp(-0.5*(n/sigma)**2)
        return (w / np.sqrt((w**2).sum()))
    elif wtype == "hann":
        w = np.hanning(L)
        return w / np.sqrt((w**2).sum())
    else:
        raise ValueError(f"Unsupported window_type={wtype}")

def _stft(x:np.ndarray, L:int, hop:int, n_fft:Optional[int], window:np.ndarray)->np.ndarray:
    N = len(x)
    if n_fft is None:
        n_fft = 1<<(int(np.ceil(np.log2(L))))
    n_frames = 1 + (N - L) // hop if N >= L else 1
    n_bins = n_fft // 2 + 1
    Z = np.empty((n_frames, n_bins), dtype=np.complex64)
    for m in range(n_frames):
        start = m*hop
        seg = np.zeros(L, dtype=float)
        if start+L <= N:
            seg[:] = x[start:start+L]
        else:
            tail = N - start
            if tail > 0:
                seg[:tail] = x[start:]
        segw = seg * window
        Z[m,:] = np.fft.rfft(segw, n=n_fft)
    return Z

def _istft(Z:np.ndarray, L:int, hop:int, n_fft:int, window:np.ndarray, length:int)->np.ndarray:
    M, K_r = Z.shape
    x_rec = np.zeros(length + L, dtype=float)
    win_acc = np.zeros(length + L, dtype=float)
    for m in range(M):
        frame = np.fft.irfft(Z[m,:], n=n_fft).real[:L]
        start = m*hop
        x_rec[start:start+L] += frame * window
        win_acc[start:start+L] += window**2
    nz = win_acc > 1e-12
    x_out = np.zeros_like(x_rec)
    x_out[nz] = x_rec[nz] / win_acc[nz]
    return x_out[:length]

def _hz_to_bin(f:float, fs:float, n_fft:int)->int:
    return int(np.clip(round(f*n_fft/fs), 0, n_fft//2))

def _default_bands(fs:float)->List[Tuple[float,float]]:
    return [(0.0, 0.02*fs), (0.02*fs, 0.15*fs), (0.15*fs, 0.5*fs)]

def _apply_band_masks(Z:np.ndarray, fs:float, n_fft:int, bands:List[Tuple[float,float]])->List[np.ndarray]:
    M, K_r = Z.shape
    outs = []
    for (f0, f1) in bands:
        b0 = _hz_to_bin(max(0.0,f0), fs, n_fft)
        b1 = _hz_to_bin(min(fs/2,f1), fs, n_fft)
        mask = np.zeros_like(Z, dtype=np.float32)
        mask[:, b0:b1+1] = 1.0
        outs.append(Z * mask)
    return outs

def _simple_ridge_mask(Z:np.ndarray, max_peaks:int)->np.ndarray:
    M, K_r = Z.shape
    A = np.abs(Z)
    mask = np.zeros_like(Z, dtype=np.float32)
    for m in range(M):
        amp = A[m]
        idx = np.argsort(amp)[-max_peaks:]
        for k in idx:
            mask[m, max(0,k-1):min(K_r, k+2)] = 1.0
    return Z * mask

def _gabor_decompose_impl(x: np.ndarray, cfg: GaborConfig) -> DecompResult:
    x = np.asarray(x, dtype=float).ravel()
    N = len(x)
    L = cfg.win_len
    hop = cfg.hop
    n_fft = cfg.n_fft or (1<<(int(np.ceil(np.log2(L)))))
    window = _make_window(L, cfg.window_type, cfg.gaussian_sigma)

    Z = _stft(x, L, hop, n_fft, window)
    fs = cfg.fs

    components: Dict[str, np.ndarray] = {}
    masks_meta = {}

    if cfg.bands is None and not cfg.ridge:
        bands = _default_bands(fs)
    else:
        bands = cfg.bands

    trend = np.zeros_like(x)
    seasonal = np.zeros_like(x)

    if bands is not None:
        band_Zs = _apply_band_masks(Z, fs, n_fft, bands)
        names = ["Trend_LF", "Seasonal_MF", "Noise_HF"] if len(bands)==3 else [f"Band_{i}" for i in range(len(bands))]
        for i, (name, Zb) in enumerate(zip(names, band_Zs)):
            xr = _istft(Zb, L, hop, n_fft, window, N)
            components[name] = xr
            if i == 0: trend += xr
            elif i == 1: seasonal += xr
            
        masks_meta["mode"] = "bands"
        masks_meta["bands"] = bands

    if cfg.ridge:
        Zr = _simple_ridge_mask(Z, cfg.ridge_max_peaks)
        xr = _istft(Zr, L, hop, n_fft, window, N)
        components["Ridge_AMFM"] = xr
        masks_meta["ridge_max_peaks"] = cfg.ridge_max_peaks
        masks_meta["mode"] = "ridge" if bands is None else "bands+ridge"
        # If ridge is used, maybe it contributes to seasonal?
        # For now, just store it.

    if components:
        s = np.zeros(N)
        for v in components.values():
            s += v
        residual = x - s
    else:
        residual = x.copy()

    meta = dict(
        fs=fs, win_len=L, hop=hop, n_fft=n_fft, window_type=cfg.window_type,
        gaussian_sigma=cfg.gaussian_sigma, tight_frame=cfg.tight_frame,
        masks=masks_meta, stft_shape=Z.shape
    )
    
    return DecompResult(
        trend=trend,
        season=seasonal,
        residual=residual,
        components=components,
        meta=meta
    )

@register("Gabor")
def gabor_decompose(x: np.ndarray, cfg: DecompositionConfig) -> DecompResult:
    # Convert cfg.params to GaborConfig
    g_cfg = GaborConfig(**{k: v for k, v in cfg.params.items() if k in GaborConfig.__annotations__})
    return _gabor_decompose_impl(x, g_cfg)
