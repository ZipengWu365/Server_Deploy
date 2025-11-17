"""
FastTimes - Python fallback implementation
This is a pure Python implementation when C++ extension is not available
"""

import numpy as np
from scipy.linalg import svd

def fastssa(X, L, r):
    """
    Fast Singular Spectrum Analysis (SSA) reconstruction
    
    Args:
        X: Input time series (n_samples, n_features) or (n_samples,)
        L: Window length for SSA
        r: Number of components to keep
    
    Returns:
        X_reconstructed: Reconstructed time series with same shape as X
    """
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    
    n_samples, n_features = X.shape
    X_reconstructed = np.zeros_like(X, dtype=np.float32)
    
    for feat_idx in range(n_features):
        series = X[:, feat_idx].astype(np.float64)
        N = len(series)
        
        # Handle edge case where series is too short
        if N < L:
            X_reconstructed[:, feat_idx] = series.astype(np.float32)
            continue
            
        K = N - L + 1
        
        # Build trajectory matrix
        trajectory = np.zeros((L, K))
        for i in range(K):
            trajectory[:, i] = series[i:i+L]
        
        # SVD decomposition
        U, S, Vt = svd(trajectory, full_matrices=False)
        
        # Reconstruct with top r components
        r_use = min(r, len(S))
        trajectory_rec = U[:, :r_use] @ np.diag(S[:r_use]) @ Vt[:r_use, :]
        
        # Diagonal averaging to get back time series
        reconstructed = np.zeros(N)
        counts = np.zeros(N)
        
        for i in range(K):
            reconstructed[i:i+L] += trajectory_rec[:, i]
            counts[i:i+L] += 1
        
        reconstructed /= counts
        X_reconstructed[:, feat_idx] = reconstructed.astype(np.float32)
    
    return X_reconstructed.squeeze() if X_reconstructed.shape[1] == 1 else X_reconstructed


def faststd(X, block_size, seasonal_rank):
    """
    Fast Seasonal-Trend Decomposition
    
    Args:
        X: Input time series (n_samples, n_features)
        block_size: Block size for local trend
        seasonal_rank: Rank for seasonal component extraction
    
    Returns:
        trend: Trend component
        seasonal: Seasonal/dispersion component  
        residual: Residual component
    """
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    
    n_samples, n_features = X.shape
    
    trend = np.zeros_like(X, dtype=np.float32)
    seasonal = np.zeros_like(X, dtype=np.float32)
    
    for feat_idx in range(n_features):
        series = X[:, feat_idx].astype(np.float64)
        
        # Extract trend using moving average
        if block_size > 1:
            window = np.ones(block_size) / block_size
            trend_component = np.convolve(series, window, mode='same')
            # Ensure exact length match
            if len(trend_component) != len(series):
                trend_component = trend_component[:len(series)]
        else:
            trend_component = series.copy()
        
        # Detrended series
        detrended = series - trend_component
        
        # Extract seasonal component using SSA
        if len(detrended) > 2 * seasonal_rank:
            L = min(len(detrended) // 2, 50)
            seasonal_component = fastssa(detrended.reshape(-1, 1), L=L, r=seasonal_rank)
            if seasonal_component.ndim > 1:
                seasonal_component = seasonal_component.ravel()
            # Ensure exact length match
            if len(seasonal_component) != len(series):
                seasonal_component = seasonal_component[:len(series)]
        else:
            seasonal_component = detrended.copy()
        
        trend[:, feat_idx] = trend_component[:len(series)].astype(np.float32)
        seasonal[:, feat_idx] = seasonal_component[:len(series)].astype(np.float32)
    
    residual = X - trend - seasonal
    
    return (
        trend.squeeze() if trend.shape[1] == 1 else trend,
        seasonal.squeeze() if seasonal.shape[1] == 1 else seasonal,
        residual.squeeze() if residual.shape[1] == 1 else residual
    )


# Make functions available at module level
__all__ = ['fastssa', 'faststd']
