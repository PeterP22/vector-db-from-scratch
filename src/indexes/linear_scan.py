"""Linear scan (brute force) vector index implementation.

This is the simplest possible nearest neighbor search - compare the query
against every vector in the database. It's O(n) complexity but guarantees
100% recall (finds exact nearest neighbors).

Used as a baseline to evaluate approximate methods.
"""

from typing import List, Optional, Dict, Any
import heapq
import numpy as np

from ..core.base import BaseIndex, SearchResult
from ..core.vector import Vector
from ..core.distance import Distance, DistanceMetric


class LinearScan(BaseIndex):
    """Brute force linear scan index.

    This index simply stores all vectors and performs exhaustive search
    by comparing the query against every vector in the database.

    Time Complexity:
        - Add: O(1)
        - Search: O(n * d) where n=num_vectors, d=dimension

    Space Complexity: O(n * d)

    Guarantees:
        - 100% recall (exact nearest neighbors)
        - No false negatives

    Use Cases:
        - Baseline for benchmarking approximate methods
        - Small datasets (< 10,000 vectors)
        - When exact results are required
    """

    def __init__(
        self,
        dimension: int,
        metric: DistanceMetric = DistanceMetric.L2
    ):
        """Initialize the linear scan index.

        Args:
            dimension: Dimensionality of vectors
            metric: Distance metric to use
        """
        super().__init__(dimension)
        self.metric = metric
        self.vectors: List[Vector] = []
        self.metadata: List[Optional[Dict[str, Any]]] = []
        self._distance_func = Distance.get_distance_function(metric)

    def add(
        self,
        vectors: List[Vector],
        metadata: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """Add vectors to the index.

        Args:
            vectors: List of vectors to add
            metadata: Optional metadata for each vector

        Raises:
            ValueError: If vector dimensions don't match
        """
        self._validate_vectors(vectors)

        self.vectors.extend(vectors)

        # Handle metadata
        if metadata is None:
            self.metadata.extend([None] * len(vectors))
        else:
            if len(metadata) != len(vectors):
                raise ValueError(
                    f"Metadata length {len(metadata)} doesn't match vectors length {len(vectors)}"
                )
            self.metadata.extend(metadata)

        self.num_vectors = len(self.vectors)

    def search(
        self,
        query: Vector,
        k: int = 10,
        **kwargs
    ) -> SearchResult:
        """Search for k nearest neighbors using linear scan.

        Args:
            query: Query vector
            k: Number of nearest neighbors to return
            **kwargs: Additional parameters (unused for linear scan)

        Returns:
            SearchResult with indices and distances of k nearest neighbors

        Raises:
            ValueError: If query dimension doesn't match or k > num_vectors
        """
        self._validate_dimension(query)

        if self.num_vectors == 0:
            return SearchResult(indices=[], distances=[], metadata=[])

        if k > self.num_vectors:
            k = self.num_vectors

        # Compute distances to all vectors
        distances = []
        for i, vector in enumerate(self.vectors):
            dist = self._distance_func(query, vector)
            distances.append((dist, i))

        # Get k smallest distances using heap (efficient for small k)
        # heapq.nsmallest returns [(dist, idx), ...] sorted by distance
        top_k = heapq.nsmallest(k, distances, key=lambda x: x[0])

        # Extract indices and distances
        result_distances = [dist for dist, _ in top_k]
        result_indices = [idx for _, idx in top_k]

        # Extract metadata if available
        result_metadata = None
        if any(m is not None for m in self.metadata):
            result_metadata = [self.metadata[idx] for idx in result_indices]

        return SearchResult(
            indices=result_indices,
            distances=result_distances,
            metadata=result_metadata
        )

    def _get_additional_stats(self) -> Dict[str, Any]:
        """Get linear scan specific statistics.

        Returns:
            Dictionary with metric information
        """
        return {
            "metric": self.metric.value,
            "search_complexity": f"O({self.num_vectors} * {self.dimension})",
            "exact_search": True
        }

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"LinearScan(dimension={self.dimension}, "
            f"num_vectors={self.num_vectors}, "
            f"metric={self.metric.value})"
        )


class OptimizedLinearScan(LinearScan):
    """Optimized linear scan using vectorized operations.

    This version uses NumPy's vectorized operations for better performance.
    Trades memory for speed by storing vectors as a single matrix.
    """

    def __init__(
        self,
        dimension: int,
        metric: DistanceMetric = DistanceMetric.L2
    ):
        """Initialize the optimized linear scan index.

        Args:
            dimension: Dimensionality of vectors
            metric: Distance metric to use
        """
        super().__init__(dimension, metric)
        self.vector_matrix: Optional[np.ndarray] = None

    def add(
        self,
        vectors: List[Vector],
        metadata: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """Add vectors to the index.

        Args:
            vectors: List of vectors to add
            metadata: Optional metadata for each vector
        """
        super().add(vectors, metadata)

        # Rebuild matrix with all vectors
        self.vector_matrix = np.vstack(self.vectors)

    def search(
        self,
        query: Vector,
        k: int = 10,
        **kwargs
    ) -> SearchResult:
        """Search using optimized vectorized operations.

        Args:
            query: Query vector
            k: Number of nearest neighbors to return
            **kwargs: Additional parameters

        Returns:
            SearchResult with k nearest neighbors
        """
        self._validate_dimension(query)

        if self.num_vectors == 0:
            return SearchResult(indices=[], distances=[], metadata=[])

        if k > self.num_vectors:
            k = self.num_vectors

        # Compute distances using vectorized operations
        if self.metric == DistanceMetric.L2:
            # Optimized L2 distance: ||a-b||^2 = ||a||^2 + ||b||^2 - 2*a·b
            distances = np.linalg.norm(self.vector_matrix - query, axis=1)
        elif self.metric == DistanceMetric.COSINE:
            # Cosine distance using vectorized operations
            query_norm = np.linalg.norm(query)
            vector_norms = np.linalg.norm(self.vector_matrix, axis=1)
            dot_products = np.dot(self.vector_matrix, query)

            # Avoid division by zero
            with np.errstate(divide='ignore', invalid='ignore'):
                similarities = dot_products / (query_norm * vector_norms)
                similarities = np.nan_to_num(similarities, nan=0.0)

            distances = 1.0 - similarities
        elif self.metric == DistanceMetric.DOT_PRODUCT:
            # Negative dot product for minimization
            distances = -np.dot(self.vector_matrix, query)
        else:
            # Fall back to loop for other metrics
            return super().search(query, k, **kwargs)

        # Get k smallest distances
        if k < self.num_vectors:
            # Use argpartition for efficiency (O(n) instead of O(n log n))
            partitioned_indices = np.argpartition(distances, k)[:k]
            top_k_indices = partitioned_indices[np.argsort(distances[partitioned_indices])]
        else:
            top_k_indices = np.argsort(distances)

        result_indices = top_k_indices.tolist()
        result_distances = distances[top_k_indices].tolist()

        # Extract metadata
        result_metadata = None
        if any(m is not None for m in self.metadata):
            result_metadata = [self.metadata[idx] for idx in result_indices]

        return SearchResult(
            indices=result_indices,
            distances=result_distances,
            metadata=result_metadata
        )

    def _get_additional_stats(self) -> Dict[str, Any]:
        """Get optimized linear scan statistics."""
        stats = super()._get_additional_stats()
        stats["optimized"] = True
        stats["vectorized"] = True
        return stats
