"""Inverted File Index (IVF) for approximate nearest neighbor search.

IVF uses clustering to partition the vector space into Voronoi cells,
enabling fast coarse-to-fine search. This is the foundation of FAISS
and many production vector databases.

Key Concepts:
1. Coarse quantization: K-means clustering creates k centroids
2. Inverted lists: Each cluster has a list of vectors (inverted index)
3. Two-stage search:
   - Coarse: Find nearest centroid(s)
   - Fine: Search vectors in those cluster(s)
4. Multi-probe: Search top-k clusters for higher recall

Algorithm:
- Build: Run k-means, assign vectors to nearest centroid
- Search: Find n_probe nearest centroids, search their inverted lists

Time Complexity:
    - Build: O(n * k * d * iterations) for k-means
    - Search: O(k * d + (n/k) * n_probe * d) where n_probe << k

Space Complexity: O(n * d + k * d)

Advantages:
- Sub-linear search time: O(n/k) per cluster
- Works well with compression (IVF+PQ is FAISS standard)
- Scalable to billions of vectors

Disadvantages:
- Recall depends on clustering quality
- Boundary vectors may be missed (mitigated by multi-probe)
- K-means build time can be slow

Production Usage:
    - FAISS uses IVF+PQ for billion-scale search
    - Often combined with product quantization for compression
"""

from typing import List, Optional, Dict, Any, Tuple
from collections import defaultdict
import numpy as np
import heapq

from ..core.base import BaseIndex, SearchResult
from ..core.vector import Vector
from ..core.distance import Distance, DistanceMetric


class KMeans:
    """K-means clustering for coarse quantization.

    This is a standard implementation of Lloyd's algorithm for k-means.
    """

    def __init__(
        self,
        n_clusters: int,
        max_iter: int = 100,
        tol: float = 1e-4,
        seed: int = 42
    ):
        """Initialize k-means clusterer.

        Args:
            n_clusters: Number of clusters (k)
            max_iter: Maximum iterations
            tol: Convergence tolerance
            seed: Random seed
        """
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.seed = seed
        self.centroids: Optional[np.ndarray] = None
        self.inertia_: float = 0.0  # Sum of squared distances to nearest centroid

    def fit(self, vectors: List[Vector]) -> 'KMeans':
        """Fit k-means to data using Lloyd's algorithm.

        Args:
            vectors: List of vectors to cluster

        Returns:
            Self
        """
        n_samples = len(vectors)
        n_features = len(vectors[0])

        # Convert to numpy array for efficiency
        X = np.vstack(vectors)

        # Initialize centroids using k-means++
        rng = np.random.RandomState(self.seed)
        centroid_indices = self._kmeans_plusplus_init(X, rng)
        self.centroids = X[centroid_indices].copy()

        # Lloyd's algorithm: iterate until convergence
        for iteration in range(self.max_iter):
            # Assign each point to nearest centroid
            labels = self._assign_labels(X)

            # Update centroids
            new_centroids = np.zeros_like(self.centroids)
            for i in range(self.n_clusters):
                cluster_points = X[labels == i]
                if len(cluster_points) > 0:
                    new_centroids[i] = cluster_points.mean(axis=0)
                else:
                    # Empty cluster: reinitialize randomly
                    new_centroids[i] = X[rng.randint(n_samples)]

            # Check convergence
            centroid_shift = np.linalg.norm(new_centroids - self.centroids, axis=1).max()
            self.centroids = new_centroids

            if centroid_shift < self.tol:
                break

        # Compute final inertia
        labels = self._assign_labels(X)
        self.inertia_ = self._compute_inertia(X, labels)

        return self

    def _kmeans_plusplus_init(self, X: np.ndarray, rng: np.random.RandomState) -> List[int]:
        """Initialize centroids using k-means++ for better convergence.

        Args:
            X: Data matrix
            rng: Random number generator

        Returns:
            List of indices of initial centroids
        """
        n_samples = X.shape[0]
        centroid_indices = []

        # Choose first centroid randomly
        centroid_indices.append(rng.randint(n_samples))

        # Choose remaining centroids with probability proportional to distance squared
        for _ in range(1, self.n_clusters):
            # Compute distances to nearest existing centroid
            distances = np.full(n_samples, np.inf)
            for idx in centroid_indices:
                dists = np.linalg.norm(X - X[idx], axis=1)
                distances = np.minimum(distances, dists)

            # Choose next centroid with probability proportional to distance squared
            probabilities = distances ** 2
            probabilities /= probabilities.sum()
            next_centroid = rng.choice(n_samples, p=probabilities)
            centroid_indices.append(next_centroid)

        return centroid_indices

    def _assign_labels(self, X: np.ndarray) -> np.ndarray:
        """Assign each point to nearest centroid.

        Args:
            X: Data matrix

        Returns:
            Array of cluster labels
        """
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        return np.argmin(distances, axis=1)

    def _compute_inertia(self, X: np.ndarray, labels: np.ndarray) -> float:
        """Compute inertia (sum of squared distances to centroids).

        Args:
            X: Data matrix
            labels: Cluster assignments

        Returns:
            Inertia value
        """
        inertia = 0.0
        for i in range(self.n_clusters):
            cluster_points = X[labels == i]
            if len(cluster_points) > 0:
                inertia += np.sum((cluster_points - self.centroids[i]) ** 2)
        return inertia

    def predict(self, vectors: List[Vector]) -> np.ndarray:
        """Predict cluster labels for vectors.

        Args:
            vectors: List of vectors

        Returns:
            Array of cluster labels
        """
        if self.centroids is None:
            raise ValueError("KMeans must be fitted before prediction")

        X = np.vstack(vectors)
        return self._assign_labels(X)

    def find_nearest_centroids(
        self,
        vector: Vector,
        n_probe: int = 1
    ) -> List[Tuple[float, int]]:
        """Find n_probe nearest centroids to a query vector.

        Args:
            vector: Query vector
            n_probe: Number of nearest centroids to return

        Returns:
            List of (distance, centroid_idx) tuples
        """
        if self.centroids is None:
            raise ValueError("KMeans must be fitted before searching")

        # Compute distances to all centroids
        distances = np.linalg.norm(self.centroids - vector, axis=1)

        # Get n_probe nearest
        n_probe = min(n_probe, self.n_clusters)
        nearest_indices = np.argpartition(distances, n_probe)[:n_probe]
        nearest_indices = nearest_indices[np.argsort(distances[nearest_indices])]

        return [(distances[i], i) for i in nearest_indices]


class IVF(BaseIndex):
    """Inverted File Index for approximate nearest neighbor search.

    This implementation uses k-means clustering to partition the space
    into Voronoi cells, then searches the inverted lists of the nearest
    clusters (multi-probe).
    """

    def __init__(
        self,
        dimension: int,
        n_clusters: int = 100,
        n_probe: int = 1,
        metric: DistanceMetric = DistanceMetric.L2,
        kmeans_max_iter: int = 100,
        seed: int = 42
    ):
        """Initialize IVF index.

        Args:
            dimension: Dimensionality of vectors
            n_clusters: Number of clusters (k). Rule of thumb: sqrt(n) to n/100
            n_probe: Number of clusters to probe during search (higher = better recall)
            metric: Distance metric to use
            kmeans_max_iter: Maximum iterations for k-means
            seed: Random seed
        """
        super().__init__(dimension)
        self.n_clusters = n_clusters
        self.n_probe = n_probe
        self.metric = metric
        self.seed = seed

        self._distance_func = Distance.get_distance_function(metric)
        self._kmeans = KMeans(n_clusters, max_iter=kmeans_max_iter, seed=seed)

        # Inverted lists: cluster_id -> list of (vector, index, metadata)
        self._inverted_lists: Dict[int, List[Tuple[Vector, int, Optional[Dict[str, Any]]]]] = defaultdict(list)
        self._is_trained = False

        # Statistics
        self._total_probes = 0
        self._total_searches = 0

    def add(
        self,
        vectors: List[Vector],
        metadata: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """Add vectors to IVF index.

        Note: This will train k-means if not already trained.

        Args:
            vectors: List of vectors to add
            metadata: Optional metadata for each vector

        Raises:
            ValueError: If vector dimensions don't match
        """
        self._validate_vectors(vectors)

        start_time = self._start_build_timer()

        if metadata is None:
            metadata = [None] * len(vectors)

        # Train k-means if not already trained
        if not self._is_trained:
            if len(vectors) < self.n_clusters:
                raise ValueError(
                    f"Need at least {self.n_clusters} vectors to train "
                    f"{self.n_clusters} clusters, got {len(vectors)}"
                )
            self._kmeans.fit(vectors)
            self._is_trained = True

        # Assign vectors to clusters and add to inverted lists
        cluster_labels = self._kmeans.predict(vectors)

        for i, (vector, label, meta) in enumerate(zip(vectors, cluster_labels, metadata)):
            vector_idx = self.num_vectors + i
            self._inverted_lists[label].append((vector, vector_idx, meta))

        self.num_vectors += len(vectors)
        self._end_build_timer(start_time)

    def search(
        self,
        query: Vector,
        k: int = 10,
        n_probe: Optional[int] = None,
        **kwargs
    ) -> SearchResult:
        """Search for k nearest neighbors using IVF.

        Algorithm:
        1. Find n_probe nearest centroids (coarse search)
        2. Search inverted lists of those clusters (fine search)
        3. Return top-k overall

        Args:
            query: Query vector
            k: Number of nearest neighbors to return
            n_probe: Number of clusters to probe (default: self.n_probe)
            **kwargs: Additional parameters

        Returns:
            SearchResult with k nearest neighbors

        Raises:
            ValueError: If index is not trained
        """
        self._validate_dimension(query)

        if not self._is_trained:
            raise ValueError("IVF index must be trained before search")

        if self.num_vectors == 0:
            return SearchResult(indices=[], distances=[], metadata=[])

        if k > self.num_vectors:
            k = self.num_vectors

        if n_probe is None:
            n_probe = self.n_probe

        # Find n_probe nearest centroids
        nearest_centroids = self._kmeans.find_nearest_centroids(query, n_probe)

        # Track statistics
        self._total_probes += n_probe
        self._total_searches += 1

        # Search inverted lists of nearest clusters
        candidates = []
        for _, cluster_id in nearest_centroids:
            inverted_list = self._inverted_lists[cluster_id]
            for vector, idx, meta in inverted_list:
                dist = self._distance_func(query, vector)
                candidates.append((dist, idx, meta))

        # Get top-k
        if not candidates:
            return SearchResult(indices=[], distances=[], metadata=[])

        if len(candidates) <= k:
            candidates.sort(key=lambda x: x[0])
            result_distances = [dist for dist, _, _ in candidates]
            result_indices = [idx for _, idx, _ in candidates]
            result_metadata = [meta for _, _, meta in candidates] if any(m is not None for _, _, m in candidates) else None
        else:
            top_k = heapq.nsmallest(k, candidates, key=lambda x: x[0])
            result_distances = [dist for dist, _, _ in top_k]
            result_indices = [idx for _, idx, _ in top_k]
            result_metadata = [meta for _, _, meta in top_k] if any(m is not None for _, _, m in top_k) else None

        return SearchResult(
            indices=result_indices,
            distances=result_distances,
            metadata=result_metadata
        )

    def get_cluster_stats(self) -> Dict[str, Any]:
        """Get statistics about cluster distribution.

        Returns:
            Dictionary with cluster statistics
        """
        cluster_sizes = [len(self._inverted_lists[i]) for i in range(self.n_clusters)]

        if not cluster_sizes or sum(cluster_sizes) == 0:
            return {
                "n_clusters": self.n_clusters,
                "empty_clusters": self.n_clusters,
                "avg_cluster_size": 0,
                "max_cluster_size": 0,
                "min_cluster_size": 0
            }

        non_empty = [s for s in cluster_sizes if s > 0]

        return {
            "n_clusters": self.n_clusters,
            "empty_clusters": sum(1 for s in cluster_sizes if s == 0),
            "avg_cluster_size": np.mean(non_empty) if non_empty else 0,
            "max_cluster_size": max(cluster_sizes),
            "min_cluster_size": min(non_empty) if non_empty else 0,
            "std_cluster_size": np.std(non_empty) if non_empty else 0
        }

    def get_search_stats(self) -> Dict[str, Any]:
        """Get statistics about search operations.

        Returns:
            Dictionary with search statistics
        """
        if self._total_searches == 0:
            return {
                "total_searches": 0,
                "avg_probes": 0,
                "avg_vectors_per_probe": 0
            }

        avg_probes = self._total_probes / self._total_searches
        avg_cluster_size = self.num_vectors / self.n_clusters if self.n_clusters > 0 else 0
        avg_vectors_per_probe = avg_probes * avg_cluster_size

        return {
            "total_searches": self._total_searches,
            "avg_probes": avg_probes,
            "avg_vectors_per_probe": avg_vectors_per_probe,
            "search_reduction": f"{(1 - avg_vectors_per_probe / self.num_vectors) * 100:.1f}%" if self.num_vectors > 0 else "0%"
        }

    def _estimate_memory(self) -> int:
        """Estimate memory usage in bytes.

        Returns:
            Approximate memory usage
        """
        # Vectors
        vectors_mem = self.num_vectors * self.dimension * 4

        # Centroids
        centroids_mem = self.n_clusters * self.dimension * 4

        # Inverted lists overhead (indices, pointers)
        inverted_lists_mem = self.num_vectors * 16  # rough estimate

        return vectors_mem + centroids_mem + inverted_lists_mem

    def _get_additional_stats(self) -> Dict[str, Any]:
        """Get IVF specific statistics."""
        stats = {
            "metric": self.metric.value,
            "n_clusters": self.n_clusters,
            "n_probe": self.n_probe,
            "is_trained": self._is_trained,
        }

        if self._is_trained:
            stats["kmeans_inertia"] = self._kmeans.inertia_
            stats.update(self.get_cluster_stats())

        if self._total_searches > 0:
            stats.update(self.get_search_stats())

        return stats

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"IVF(dimension={self.dimension}, "
            f"num_vectors={self.num_vectors}, "
            f"n_clusters={self.n_clusters}, "
            f"n_probe={self.n_probe}, "
            f"metric={self.metric.value}, "
            f"trained={self._is_trained})"
        )
