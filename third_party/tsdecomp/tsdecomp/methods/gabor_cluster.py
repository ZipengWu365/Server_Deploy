from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Sequence
from ..core import DecompositionConfig, DecompResult
from ..registry import register
from .gabor import _make_window, _stft, _istft

try:
    import faiss
except ImportError:
    faiss = None

@dataclass
class GaborClusterConfig:
    fs: float = 1.0
    win_len: int = 256
    hop: int = 64
    n_fft: Optional[int] = None
    window_type: str = "gaussian"
    gaussian_sigma: Optional[float] = None
    use_log_amp: bool = True
    n_clusters: int = 5
    max_atoms: Optional[int] = 100000
    n_iter: int = 20
    random_state: int = 42
    verbose: bool = False

@dataclass
class GaborClusterModel:
    centroids: np.ndarray
    mu: np.ndarray
    sigma: np.ndarray
    cfg: GaborClusterConfig
    
    @classmethod
    def load(cls, path: str):
        # Placeholder for loading model
        data = np.load(path, allow_pickle=True)
        return cls(centroids=data['centroids'], mu=data['mu'], sigma=data['sigma'], cfg=data['cfg'])

def _extract_gabor_features(x: np.ndarray, cfg: GaborClusterConfig, window: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float).ravel()
    N = len(x)
    L = cfg.win_len
    hop = cfg.hop
    n_fft = cfg.n_fft or (1 << int(np.ceil(np.log2(L))))
    Z = _stft(x, L, hop, n_fft, window)
    M, K_r = Z.shape

    amp = np.abs(Z)
    if cfg.use_log_amp:
        amp_feat = np.log1p(amp)
    else:
        amp_feat = amp

    if M > 1: t_idx = np.linspace(0.0, 1.0, M)
    else: t_idx = np.array([0.0])
    if K_r > 1: f_idx = np.linspace(0.0, 1.0, K_r)
    else: f_idx = np.array([0.0])

    T, F = np.meshgrid(t_idx, f_idx, indexing="ij")
    feats = np.stack([T.ravel().astype(np.float32), F.ravel().astype(np.float32), amp_feat.ravel().astype(np.float32)], axis=1)
    return feats, Z

def _assign_clusters_faiss(feats: np.ndarray, model: GaborClusterModel) -> np.ndarray:
    if faiss is None:
        raise ImportError("faiss is required for Gabor Cluster decomposition.")
    
    X = (feats - model.mu) / model.sigma
    X = X.astype(np.float32)
    d = X.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(model.centroids.astype(np.float32))
    D, I = index.search(X, 1)
    return I.ravel()

def _gabor_cluster_decompose_impl(x: np.ndarray, model: GaborClusterModel, max_clusters: Optional[int] = None) -> DecompResult:
    cfg = model.cfg
    x = np.asarray(x, dtype=float).ravel()
    N = len(x)
    L = cfg.win_len
    hop = cfg.hop
    n_fft = cfg.n_fft or (1 << int(np.ceil(np.log2(L))))
    window = _make_window(L, cfg.window_type, cfg.gaussian_sigma)

    feats, Z = _extract_gabor_features(x, cfg, window)
    labels = _assign_clusters_faiss(feats, model)

    M, K_r = Z.shape
    K = model.centroids.shape[0]
    labels_2d = labels.reshape(M, K_r)

    amp = np.abs(Z)
    energy_per_cluster = np.zeros(K, dtype=float)
    for j in range(K):
        mask = (labels_2d == j)
        if np.any(mask):
            energy_per_cluster[j] = (amp[mask] ** 2).sum()

    if max_clusters is not None and max_clusters < K:
        keep_idx = np.argsort(energy_per_cluster)[-max_clusters:]
        keep_mask = np.zeros(K, dtype=bool)
        keep_mask[keep_idx] = True
    else:
        keep_mask = np.ones(K, dtype=bool)

    components: Dict[str, np.ndarray] = {}
    used_clusters = []

    trend = np.zeros_like(x)
    seasonal = np.zeros_like(x)
    
    # Heuristic: Low freq clusters -> Trend? 
    # For now, just put everything in components and let user decide or sum all to seasonal?
    # The original code uses gabor_components_to_TS to separate trend/seasonal based on freq.
    # I'll implement a simple version here or rely on post-processing.
    
    for j in range(K):
        if not keep_mask[j]: continue
        mask = (labels_2d == j).astype(np.float32)
        if not np.any(mask): continue
        Zj = Z * mask
        xj = _istft(Zj, L, hop, n_fft, window, N)
        components[f"Cluster_{j}"] = xj
        used_clusters.append(j)
        seasonal += xj # Default to seasonal

    if components:
        sum_comp = np.zeros_like(x)
        for v in components.values(): sum_comp += v
        residual = x - sum_comp
    else:
        residual = x.copy()

    meta = dict(
        fs=cfg.fs, win_len=L, hop=hop, n_fft=n_fft, window_type=cfg.window_type,
        gaussian_sigma=cfg.gaussian_sigma, n_clusters=model.centroids.shape[0],
        used_clusters=used_clusters, max_clusters=max_clusters, feature_dim=model.centroids.shape[1]
    )
    
    return DecompResult(trend=trend, season=seasonal, residual=residual, components=components, meta=meta)

@register("GaborCluster")
def gabor_cluster_decompose(x: np.ndarray, cfg: DecompositionConfig) -> DecompResult:
    # This requires a pre-trained model. 
    # We expect the model to be passed in cfg.params['model'] or loaded from 'model_path'
    model = cfg.params.get("model")
    if model is None:
        model_path = cfg.params.get("model_path")
        if model_path:
            model = GaborClusterModel.load(model_path)
        else:
            raise ValueError("GaborCluster requires 'model' or 'model_path' in params.")
            
    max_clusters = cfg.params.get("max_clusters")
    return _gabor_cluster_decompose_impl(x, model, max_clusters)
