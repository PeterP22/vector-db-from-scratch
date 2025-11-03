"""Scalar Quantization for vector compression.

Scalar quantization compresses float32 vectors to uint8 by mapping each
dimension's range [min, max] to [0, 255]. This achieves 4x compression.

Key Concepts:
1. Per-dimension quantization: each dimension has its own min/max
2. Linear mapping: value → uint8 = (value - min) * 255 / (max - min)
3. Dequantization: uint8 → value = uint8 * (max - min) / 255 + min
4. Distance approximation: compute distances directly on quantized vectors

Algorithm:
- Train: Compute min/max for each dimension from training data
- Encode: Map float32 → uint8 using linear transformation
- Search: Compute distances on uint8 vectors (much faster!)

Benefits:
- 4x memory compression (float32 → uint8)
- 4x faster distance computation with SIMD
- Minimal accuracy loss (~95% recall maintained)
- Simple and fast to implement

Trade-offs:
- Requires training data to compute min/max
- Loses precision (quantization error)
- Not suitable for very sparse or skewed data

Production Usage:
    - Qdrant, Elasticsearch, pgvector use scalar quantization
    - Often combined with HNSW or IVF for production systems
    - Can achieve 4x+ speedup with SIMD optimizations
"""

from typing import List, Optional, Tuple
import numpy as np
from dataclasses import dataclass

from ..core.vector import Vector


@dataclass
class QuantizationParams:
    """Parameters for scalar quantization.

    Attributes:
        min_vals: Minimum value per dimension
        max_vals: Maximum value per dimension
        scale: Scale factor per dimension = 255 / (max - min)
        offset: Offset per dimension = min
    """

    min_vals: np.ndarray
    max_vals: np.ndarray
    scale: np.ndarray
    offset: np.ndarray


class ScalarQuantizer:
    """Scalar quantization: float32 → uint8 compression.

    This quantizer uses linear per-dimension quantization to compress
    vectors from 32-bit floats to 8-bit unsigned integers, achieving
    4x compression with minimal accuracy loss.
    """

    def __init__(self, dimension: int):
        """Initialize scalar quantizer.

        Args:
            dimension: Dimensionality of vectors
        """
        self.dimension = dimension
        self.params: Optional[QuantizationParams] = None
        self._is_trained = False

    def train(self, vectors: List[Vector]) -> None:
        """Train quantizer by computing min/max per dimension.

        Args:
            vectors: Training vectors to compute statistics from

        Note:
            Training computes the min/max for each dimension across
            all training vectors. These are used for quantization.
        """
        if not vectors:
            raise ValueError("Need at least one vector for training")

        # Stack vectors into matrix
        X = np.vstack(vectors)

        # Compute min/max per dimension
        min_vals = X.min(axis=0).astype(np.float32)
        max_vals = X.max(axis=0).astype(np.float32)

        # Avoid division by zero for constant dimensions
        range_vals = max_vals - min_vals
        range_vals = np.where(range_vals == 0, 1.0, range_vals)

        # Compute scale and offset
        scale = 255.0 / range_vals
        offset = min_vals

        self.params = QuantizationParams(
            min_vals=min_vals,
            max_vals=max_vals,
            scale=scale,
            offset=offset
        )
        self._is_trained = True

    def encode(self, vector: Vector) -> np.ndarray:
        """Quantize a float32 vector to uint8.

        Formula: uint8 = clip((value - offset) * scale, 0, 255)

        Args:
            vector: Float32 vector to quantize

        Returns:
            Uint8 quantized vector

        Raises:
            ValueError: If quantizer is not trained
        """
        if not self._is_trained:
            raise ValueError("Quantizer must be trained before encoding")

        # Apply linear transformation
        quantized = (vector - self.params.offset) * self.params.scale

        # Clip to [0, 255] and convert to uint8
        quantized = np.clip(quantized, 0, 255).astype(np.uint8)

        return quantized

    def decode(self, quantized: np.ndarray) -> Vector:
        """Dequantize a uint8 vector back to float32.

        Formula: value = uint8 / scale + offset

        Args:
            quantized: Uint8 quantized vector

        Returns:
            Float32 reconstructed vector

        Raises:
            ValueError: If quantizer is not trained
        """
        if not self._is_trained:
            raise ValueError("Quantizer must be trained before decoding")

        # Reverse linear transformation
        decoded = quantized.astype(np.float32) / self.params.scale + self.params.offset

        return decoded

    def encode_batch(self, vectors: List[Vector]) -> np.ndarray:
        """Quantize multiple vectors efficiently.

        Args:
            vectors: List of float32 vectors

        Returns:
            Matrix of uint8 quantized vectors (n_vectors × dimension)
        """
        if not self._is_trained:
            raise ValueError("Quantizer must be trained before encoding")

        X = np.vstack(vectors)
        quantized = (X - self.params.offset) * self.params.scale
        quantized = np.clip(quantized, 0, 255).astype(np.uint8)

        return quantized

    def decode_batch(self, quantized: np.ndarray) -> List[Vector]:
        """Dequantize multiple vectors efficiently.

        Args:
            quantized: Matrix of uint8 quantized vectors

        Returns:
            List of float32 reconstructed vectors
        """
        if not self._is_trained:
            raise ValueError("Quantizer must be trained before decoding")

        decoded = quantized.astype(np.float32) / self.params.scale + self.params.offset
        return [decoded[i] for i in range(len(decoded))]

    def l2_distance_quantized(self, q1: np.ndarray, q2: np.ndarray) -> float:
        """Compute L2 distance between two quantized vectors.

        This computes the distance directly on uint8 vectors,
        which is much faster than dequantizing first.

        Args:
            q1: First quantized vector (uint8)
            q2: Second quantized vector (uint8)

        Returns:
            Approximate L2 distance

        Note:
            This is an approximation. For exact distance, dequantize first.
            But the approximation is very good and 4x faster!
        """
        # Compute in int32 to avoid overflow
        diff = q1.astype(np.int32) - q2.astype(np.int32)

        # Scale difference back to original space
        diff_scaled = diff.astype(np.float32) / self.params.scale

        # L2 distance
        return float(np.sqrt(np.sum(diff_scaled ** 2)))

    def dot_product_quantized(self, q1: np.ndarray, q2: np.ndarray) -> float:
        """Compute dot product between two quantized vectors.

        Args:
            q1: First quantized vector (uint8)
            q2: Second quantized vector (uint8)

        Returns:
            Approximate dot product
        """
        # Decode to compute dot product accurately
        # (direct computation on uint8 has too much error)
        v1 = self.decode(q1)
        v2 = self.decode(q2)
        return float(np.dot(v1, v2))

    def get_compression_ratio(self) -> float:
        """Get compression ratio achieved by quantization.

        Returns:
            Compression ratio (typically 4.0 for float32 → uint8)
        """
        return 4.0  # float32 (4 bytes) → uint8 (1 byte)

    def get_memory_overhead(self) -> int:
        """Get memory overhead for storing quantization parameters.

        Returns:
            Memory overhead in bytes
        """
        # 4 arrays of float32: min, max, scale, offset
        return 4 * self.dimension * 4

    def __repr__(self) -> str:
        """String representation."""
        trained_str = "trained" if self._is_trained else "untrained"
        return f"ScalarQuantizer(dimension={self.dimension}, {trained_str})"


def compute_quantization_error(
    original: List[Vector],
    quantizer: ScalarQuantizer
) -> Tuple[float, float, float]:
    """Compute quantization error statistics.

    Args:
        original: Original float32 vectors
        quantizer: Trained quantizer

    Returns:
        Tuple of (mean_error, max_error, relative_error)
    """
    errors = []

    for vector in original:
        # Quantize and dequantize
        quantized = quantizer.encode(vector)
        reconstructed = quantizer.decode(quantized)

        # Compute L2 error
        error = np.linalg.norm(vector - reconstructed)
        errors.append(error)

    mean_error = np.mean(errors)
    max_error = np.max(errors)

    # Relative error (normalized by vector magnitude)
    magnitudes = [np.linalg.norm(v) for v in original]
    relative_error = mean_error / np.mean(magnitudes) if np.mean(magnitudes) > 0 else 0

    return mean_error, max_error, relative_error


def demonstrate_compression():
    """Demonstrate compression ratio and accuracy."""
    print("Scalar Quantization Demo")
    print("=" * 50)

    # Generate random vectors
    dimension = 128
    n_vectors = 1000

    vectors = [np.random.randn(dimension).astype(np.float32) for _ in range(n_vectors)]

    # Train quantizer
    quantizer = ScalarQuantizer(dimension)
    quantizer.train(vectors)

    # Compute memory savings
    original_size = n_vectors * dimension * 4  # float32
    quantized_size = n_vectors * dimension * 1  # uint8
    overhead = quantizer.get_memory_overhead()
    total_quantized_size = quantized_size + overhead

    print(f"\nMemory Usage:")
    print(f"  Original (float32): {original_size / 1024:.2f} KB")
    print(f"  Quantized (uint8): {quantized_size / 1024:.2f} KB")
    print(f"  Overhead (params): {overhead / 1024:.2f} KB")
    print(f"  Total quantized: {total_quantized_size / 1024:.2f} KB")
    print(f"  Compression ratio: {original_size / total_quantized_size:.2f}x")

    # Compute quantization error
    mean_err, max_err, rel_err = compute_quantization_error(vectors[:100], quantizer)

    print(f"\nQuantization Error (100 vectors):")
    print(f"  Mean L2 error: {mean_err:.4f}")
    print(f"  Max L2 error: {max_err:.4f}")
    print(f"  Relative error: {rel_err:.2%}")

    # Test distance preservation
    print(f"\nDistance Preservation Test:")
    v1, v2 = vectors[0], vectors[1]
    q1, q2 = quantizer.encode(v1), quantizer.encode(v2)

    original_dist = float(np.linalg.norm(v1 - v2))
    quantized_dist = quantizer.l2_distance_quantized(q1, q2)

    print(f"  Original distance: {original_dist:.4f}")
    print(f"  Quantized distance: {quantized_dist:.4f}")
    print(f"  Relative error: {abs(original_dist - quantized_dist) / original_dist:.2%}")


if __name__ == "__main__":
    demonstrate_compression()
