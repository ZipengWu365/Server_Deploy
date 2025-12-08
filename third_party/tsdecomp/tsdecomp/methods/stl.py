import numpy as np
from ..core import DecompositionConfig, DecompResult
from ..registry import register

@register("STL")
def stl_decompose(x: np.ndarray, cfg: DecompositionConfig) -> DecompResult:
    """
    STL decomposition using statsmodels.
    """
    try:
        from statsmodels.tsa.seasonal import STL
    except ImportError as exc:
        raise ImportError("statsmodels is required for STL decomposition.") from exc

    period = cfg.params.get("period")
    if period is None:
        # Try to infer or default? 
        # For now, raise error as per original code
        raise ValueError("STL requires 'period' in params.")
    period = int(period)

    # Extract other params for STL
    stl_kwargs = {k: v for k, v in cfg.params.items() if k != "period"}
    
    stl = STL(x, period=period, **stl_kwargs)
    res = stl.fit()

    return DecompResult(
        trend=res.trend,
        season=res.seasonal,
        residual=res.resid,
        meta={"params": cfg.params}
    )
