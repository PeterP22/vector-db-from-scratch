"""Vector quantization implementations."""

from .scalar import ScalarQuantizer, QuantizationParams, compute_quantization_error
from .simd_ops import SIMDDistanceComputer
from .product import ProductQuantizer, ProductQuantizerParams, compute_pq_error

__all__ = [
    "ScalarQuantizer",
    "QuantizationParams",
    "compute_quantization_error",
    "SIMDDistanceComputer",
    "ProductQuantizer",
    "ProductQuantizerParams",
    "compute_pq_error",
]
