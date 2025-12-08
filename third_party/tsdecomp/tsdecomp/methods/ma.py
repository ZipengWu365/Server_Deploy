import numpy as np
from typing import Optional, List, Dict, Any
from ..core import DecompositionConfig, DecompResult
from ..registry import register

def _moving_average(y: np.ndarray, window: int) -> np.ndarray:
    window = max(1, int(window))
    if window == 1 or len(y) == 0:
        return y.copy()
    kernel = np.ones(window) / window
    return np.convolve(y, kernel, mode="same")

def _estimate_seasonal_indices(detrended: np.ndarray, period: int) -> np.ndarray:
    period = max(1, int(period))
    season = np.zeros_like(detrended)
    for offset in range(period):
        idx = np.arange(offset, len(detrended), period)
        if idx.size == 0: continue
        mean_val = np.mean(detrended[idx])
        season[idx] = mean_val
    season -= np.mean(season)
    return season

@register("MA")
def ma_decompose(x: np.ndarray, cfg: DecompositionConfig) -> DecompResult:
    y = np.asarray(x, dtype=float)
    default_window = max(3, len(y) // 20)
    trend_window = int(cfg.params.get("trend_window", default_window))
    if trend_window % 2 == 0: trend_window += 1
    
    trend = _moving_average(y, trend_window)

    season_period = cfg.params.get("season_period")
    if season_period:
        season = _estimate_seasonal_indices(y - trend, int(season_period))
    else:
        season = np.zeros_like(y)
        
    residual = y - trend - season
    
    return DecompResult(trend=trend, season=season, residual=residual, meta={"params": cfg.params})
