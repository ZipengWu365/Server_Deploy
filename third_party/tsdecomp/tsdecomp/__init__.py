from .core import DecompositionConfig, DecompResult, decompose, batch_decompose
from .registry import register, METHODS
# Import methods to ensure registration
from . import methods

__version__ = "0.1.0"
