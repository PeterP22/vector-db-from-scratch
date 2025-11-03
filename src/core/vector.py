"""Core vector operations and utilities."""

from typing import List, Union
import numpy as np
from numpy.typing import NDArray


# Type alias for vectors
Vector = NDArray[np.float32]


class VectorOps:
    """Utility class for common vector operations.

    This class provides optimized implementations of vector operations
    used throughout the vector database implementations.
    """

    @staticmethod
    def normalize(v: Vector) -> Vector:
        """Normalize a vector to unit length.

        Args:
            v: Input vector

        Returns:
            Normalized vector (L2 norm = 1)

        Note:
            Returns original vector if norm is 0 to avoid division by zero.
        """
        norm = np.linalg.norm(v)
        if norm == 0:
            return v
        return v / norm

    @staticmethod
    def magnitude(v: Vector) -> float:
        """Calculate the L2 (Euclidean) magnitude of a vector.

        Args:
            v: Input vector

        Returns:
            L2 norm (magnitude) of the vector
        """
        return float(np.linalg.norm(v))

    @staticmethod
    def dot(v1: Vector, v2: Vector) -> float:
        """Calculate dot product of two vectors.

        Args:
            v1: First vector
            v2: Second vector

        Returns:
            Dot product (inner product) of v1 and v2
        """
        return float(np.dot(v1, v2))

    @staticmethod
    def add(v1: Vector, v2: Vector) -> Vector:
        """Add two vectors element-wise.

        Args:
            v1: First vector
            v2: Second vector

        Returns:
            Element-wise sum of v1 and v2
        """
        return v1 + v2

    @staticmethod
    def subtract(v1: Vector, v2: Vector) -> Vector:
        """Subtract two vectors element-wise.

        Args:
            v1: First vector
            v2: Second vector

        Returns:
            Element-wise difference (v1 - v2)
        """
        return v1 - v2

    @staticmethod
    def scale(v: Vector, scalar: float) -> Vector:
        """Scale a vector by a scalar value.

        Args:
            v: Input vector
            scalar: Scaling factor

        Returns:
            Scaled vector (v * scalar)
        """
        return v * scalar

    @staticmethod
    def validate_dimensions(vectors: List[Vector]) -> bool:
        """Validate that all vectors have the same dimensionality.

        Args:
            vectors: List of vectors to validate

        Returns:
            True if all vectors have same dimension, False otherwise
        """
        if not vectors:
            return True

        first_dim = len(vectors[0])
        return all(len(v) == first_dim for v in vectors)

    @staticmethod
    def to_float32(v: Union[List[float], NDArray]) -> Vector:
        """Convert vector to float32 numpy array.

        Args:
            v: Input vector (list or numpy array)

        Returns:
            Vector as float32 numpy array
        """
        return np.array(v, dtype=np.float32)

    @staticmethod
    def batch_normalize(vectors: List[Vector]) -> List[Vector]:
        """Normalize a batch of vectors to unit length.

        Args:
            vectors: List of vectors to normalize

        Returns:
            List of normalized vectors
        """
        return [VectorOps.normalize(v) for v in vectors]

    @staticmethod
    def random_vector(dimension: int, normalized: bool = False) -> Vector:
        """Generate a random vector.

        Args:
            dimension: Dimensionality of the vector
            normalized: Whether to normalize the vector to unit length

        Returns:
            Random vector with specified dimension
        """
        v = np.random.randn(dimension).astype(np.float32)
        if normalized:
            v = VectorOps.normalize(v)
        return v

    @staticmethod
    def zero_vector(dimension: int) -> Vector:
        """Create a zero vector of specified dimension.

        Args:
            dimension: Dimensionality of the vector

        Returns:
            Zero vector
        """
        return np.zeros(dimension, dtype=np.float32)


def print_vector_stats(vectors: List[Vector], name: str = "Dataset") -> None:
    """Print statistics about a collection of vectors.

    Args:
        vectors: List of vectors
        name: Name of the dataset for display
    """
    if not vectors:
        print(f"{name}: Empty dataset")
        return

    dims = len(vectors[0])
    count = len(vectors)

    magnitudes = [VectorOps.magnitude(v) for v in vectors]
    avg_mag = np.mean(magnitudes)
    min_mag = np.min(magnitudes)
    max_mag = np.max(magnitudes)

    print(f"{name} Statistics:")
    print(f"  Count: {count}")
    print(f"  Dimensions: {dims}")
    print(f"  Magnitude - Mean: {avg_mag:.4f}, Min: {min_mag:.4f}, Max: {max_mag:.4f}")
