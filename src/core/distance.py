"""Distance metrics for vector similarity search.

This module implements various distance metrics used in vector search:
- L2 (Euclidean) distance: sqrt(sum((a[i] - b[i])^2))
- Cosine similarity/distance: 1 - (dot(a,b) / (||a|| * ||b||))
- Dot product (MIPS): dot(a, b)

Note on choosing metrics:
- L2: Best for absolute distance in space (embeddings, coordinates)
- Cosine: Best when direction matters more than magnitude (text, documents)
- Dot product: Best for maximum inner product search (recommendation systems)
"""

from enum import Enum
from typing import Callable, List, Tuple
import numpy as np

from .vector import Vector, VectorOps


class DistanceMetric(Enum):
    """Available distance metrics for vector similarity."""

    L2 = "l2"  # Euclidean distance
    COSINE = "cosine"  # Cosine distance (1 - cosine similarity)
    DOT_PRODUCT = "dot_product"  # Negative dot product (for minimization)
    MANHATTAN = "manhattan"  # L1 distance (Manhattan/taxicab)


class Distance:
    """Distance calculation functions for vector similarity search.

    All distance functions follow the convention that smaller values
    indicate more similar vectors (for consistency in nearest neighbor search).
    """

    @staticmethod
    def l2_distance(v1: Vector, v2: Vector) -> float:
        """Calculate L2 (Euclidean) distance between two vectors.

        Formula: sqrt(sum((a[i] - b[i])^2))

        Args:
            v1: First vector
            v2: Second vector

        Returns:
            L2 distance (always >= 0)

        Note:
            This is the "straight-line" distance in n-dimensional space.
            Common for embeddings where absolute position matters.
        """
        diff = v1 - v2
        return float(np.sqrt(np.dot(diff, diff)))

    @staticmethod
    def l2_distance_squared(v1: Vector, v2: Vector) -> float:
        """Calculate squared L2 distance (avoids sqrt for efficiency).

        Formula: sum((a[i] - b[i])^2)

        Args:
            v1: First vector
            v2: Second vector

        Returns:
            Squared L2 distance

        Note:
            Faster than l2_distance since it avoids sqrt.
            Preserves ordering (useful for nearest neighbor search).
        """
        diff = v1 - v2
        return float(np.dot(diff, diff))

    @staticmethod
    def cosine_similarity(v1: Vector, v2: Vector) -> float:
        """Calculate cosine similarity between two vectors.

        Formula: dot(a, b) / (||a|| * ||b||)

        Args:
            v1: First vector
            v2: Second vector

        Returns:
            Cosine similarity in range [-1, 1]
            1 = same direction, 0 = orthogonal, -1 = opposite direction

        Note:
            Measures angle between vectors, ignoring magnitude.
            Returns 0 if either vector is zero.
        """
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(np.dot(v1, v2) / (norm1 * norm2))

    @staticmethod
    def cosine_distance(v1: Vector, v2: Vector) -> float:
        """Calculate cosine distance between two vectors.

        Formula: 1 - cosine_similarity(a, b)

        Args:
            v1: First vector
            v2: Second vector

        Returns:
            Cosine distance in range [0, 2]
            0 = same direction, 1 = orthogonal, 2 = opposite direction

        Note:
            Converted to distance for consistency (smaller = more similar).
            Commonly used for text embeddings and document similarity.
        """
        return 1.0 - Distance.cosine_similarity(v1, v2)

    @staticmethod
    def dot_product(v1: Vector, v2: Vector) -> float:
        """Calculate negative dot product (for similarity search).

        Formula: -dot(a, b)

        Args:
            v1: First vector
            v2: Second vector

        Returns:
            Negative dot product (negated for minimization)

        Note:
            Negated so smaller values = more similar (consistent with other metrics).
            Used for Maximum Inner Product Search (MIPS).
            Useful for recommendation systems, when vectors are already normalized.
        """
        return -float(np.dot(v1, v2))

    @staticmethod
    def dot_product_similarity(v1: Vector, v2: Vector) -> float:
        """Calculate dot product similarity (higher = more similar).

        Formula: dot(a, b)

        Args:
            v1: First vector
            v2: Second vector

        Returns:
            Dot product (positive for similarity)

        Note:
            This is the raw dot product (not negated).
            Use when you want higher values to indicate more similarity.
        """
        return float(np.dot(v1, v2))

    @staticmethod
    def manhattan_distance(v1: Vector, v2: Vector) -> float:
        """Calculate Manhattan (L1) distance between two vectors.

        Formula: sum(|a[i] - b[i]|)

        Args:
            v1: First vector
            v2: Second vector

        Returns:
            Manhattan distance (always >= 0)

        Note:
            Also called "taxicab" or "city block" distance.
            Sum of absolute differences along each dimension.
        """
        return float(np.sum(np.abs(v1 - v2)))

    @staticmethod
    def get_distance_function(metric: DistanceMetric) -> Callable[[Vector, Vector], float]:
        """Get the distance function for a given metric.

        Args:
            metric: Distance metric to use

        Returns:
            Distance function that takes two vectors and returns a float

        Raises:
            ValueError: If metric is not supported
        """
        if metric == DistanceMetric.L2:
            return Distance.l2_distance
        elif metric == DistanceMetric.COSINE:
            return Distance.cosine_distance
        elif metric == DistanceMetric.DOT_PRODUCT:
            return Distance.dot_product
        elif metric == DistanceMetric.MANHATTAN:
            return Distance.manhattan_distance
        else:
            raise ValueError(f"Unknown distance metric: {metric}")


def batch_pairwise_distances(
    queries: List[Vector],
    dataset: List[Vector],
    metric: DistanceMetric = DistanceMetric.L2
) -> np.ndarray:
    """Compute pairwise distances between queries and dataset vectors.

    Args:
        queries: List of query vectors (m vectors)
        dataset: List of dataset vectors (n vectors)
        metric: Distance metric to use

    Returns:
        Matrix of shape (m, n) where element [i,j] is distance from
        query i to dataset vector j

    Note:
        This is optimized for batch computation when possible.
    """
    m = len(queries)
    n = len(dataset)
    distances = np.zeros((m, n), dtype=np.float32)

    dist_func = Distance.get_distance_function(metric)

    for i, query in enumerate(queries):
        for j, data_vec in enumerate(dataset):
            distances[i, j] = dist_func(query, data_vec)

    return distances


def compare_metrics_demo(v1: Vector, v2: Vector) -> None:
    """Demonstrate different distance metrics on two vectors.

    Args:
        v1: First vector
        v2: Second vector
    """
    print(f"Vector 1: {v1[:4]}... (showing first 4 dims)")
    print(f"Vector 2: {v2[:4]}... (showing first 4 dims)")
    print(f"\nDistance Metrics:")
    print(f"  L2 distance:        {Distance.l2_distance(v1, v2):.4f}")
    print(f"  L2 squared:         {Distance.l2_distance_squared(v1, v2):.4f}")
    print(f"  Cosine similarity:  {Distance.cosine_similarity(v1, v2):.4f}")
    print(f"  Cosine distance:    {Distance.cosine_distance(v1, v2):.4f}")
    print(f"  Dot product (sim):  {Distance.dot_product_similarity(v1, v2):.4f}")
    print(f"  Manhattan distance: {Distance.manhattan_distance(v1, v2):.4f}")

    # Show with normalized vectors
    v1_norm = VectorOps.normalize(v1)
    v2_norm = VectorOps.normalize(v2)
    print(f"\nWith normalized vectors:")
    print(f"  L2 distance:        {Distance.l2_distance(v1_norm, v2_norm):.4f}")
    print(f"  Cosine distance:    {Distance.cosine_distance(v1_norm, v2_norm):.4f}")
    print(f"  Dot product (sim):  {Distance.dot_product_similarity(v1_norm, v2_norm):.4f}")
