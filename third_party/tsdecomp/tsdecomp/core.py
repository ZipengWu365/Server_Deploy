from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Union, List
import numpy as np
from .registry import METHODS

@dataclass
class DecompositionConfig:
    method: str
    params: Dict[str, Any] = field(default_factory=dict)
    return_components: bool = True
    seed: Optional[int] = None

@dataclass
class DecompResult:
    trend: np.ndarray
    season: np.ndarray
    residual: np.ndarray
    components: Dict[str, np.ndarray] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)

def decompose(x: np.ndarray, cfg: DecompositionConfig) -> DecompResult:
    """
    Decompose a time series using the specified method and configuration.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError("Input time series must be 1D array")
    
    if cfg.method not in METHODS:
        raise ValueError(f"Unknown decomposition method: {cfg.method}. Available: {list(METHODS.keys())}")
    
    method_fn = METHODS[cfg.method]
    return method_fn(x, cfg)

def batch_decompose(xs: List[np.ndarray], cfg: DecompositionConfig) -> List[DecompResult]:
    """
    Decompose a batch of time series.
    """
    return [decompose(x, cfg) for x in xs]
