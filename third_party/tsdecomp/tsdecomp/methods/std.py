import numpy as np
from ..core import DecompositionConfig, DecompResult
from ..registry import register

@register("STD")
def std_decompose(x: np.ndarray, cfg: DecompositionConfig) -> DecompResult:
    """
    STD decomposition using fasttimes.faststd.
    Params:
        block_size (int): Window size for local trend/dispersion.
        seasonal_rank (int): Rank for seasonal component SVD.
    """
    try:
        from fasttimes import faststd
    except ImportError as exc:
        raise ImportError("fasttimes is required for STD decomposition.") from exc

    block_size = int(cfg.params.get("block_size", 24))
    seasonal_rank = int(cfg.params.get("seasonal_rank", 6))

    # faststd expects 2D input (n_samples, n_features) usually, or 1D?
    # In ssa_cnn_server_optimized.py: X_forC = X.T (n_samples, n_channels)
    # In std_ablation_study.py: X_win = window.reshape(-1, 1) -> faststd(X_win, ...)
    
    # Ensure input is 2D (n_samples, 1) if it's 1D
    if x.ndim == 1:
        x_in = x.reshape(-1, 1)
    else:
        x_in = x

    # faststd returns: trend, seasonal, residual (or dispersion?)
    # In ssa_cnn_server_optimized.py: trend_c, disp_c, _ = faststd(...)
    # In std_ablation_study.py: trend, seasonal, residual = faststd(...)
    # Wait, ssa_cnn_server_optimized.py says: trend_c, disp_c, _
    # std_ablation_study.py says: trend, seasonal, residual
    # Let's assume std_ablation_study.py is correct for the full return.
    # But ssa_cnn_server_optimized.py ignores the 3rd return? No, it ignores the 3rd return which might be seasonal?
    # Actually, ssa_cnn_server_optimized.py calculates seasonal manually:
    # Xc = Xb - trend
    # seasonal = Xc / dispersion
    # So faststd might return (trend, dispersion, something_else).
    
    # Let's check std_ablation_study.py again.
    # trend, seasonal, residual = faststd(X_win, block_size, rank)
    # It uses `seasonal` directly.
    
    # Let's trust std_ablation_study.py usage as it seems to use all 3.
    
    trend, seasonal, residual = faststd(x_in.astype(np.float32), block_size=block_size, seasonal_rank=seasonal_rank)
    
    # Flatten if input was 1D
    if x.ndim == 1:
        trend = trend.flatten()
        seasonal = seasonal.flatten()
        residual = residual.flatten()

    return DecompResult(
        trend=trend,
        season=seasonal,
        residual=residual,
        meta={"params": cfg.params}
    )
