"""Locality Sensitive Hashing (LSH) for approximate nearest neighbor search.

LSH is a probabilistic technique that hashes similar items into the same buckets
with high probability. Unlike traditional hashing, hash COLLISIONS ARE DESIRED.

Key Concepts:
- Random Projection (SimHash): Use random hyperplanes to partition space
- Hash function: h(v) = sign(v · r) where r is a random unit vector
- Multi-table strategy: Use L tables with different random projections
- Trade-off: More tables/bits = higher recall but more memory/time

Algorithm:
1. Generate L hash tables, each with k random hyperplanes
2. For each vector, compute k-bit hash code per table
3. Store vector in buckets indexed by hash codes
4. Query: Compute hash codes, retrieve candidates from all buckets, rank by distance

Advantages:
- Works well in high dimensions (where KD-Tree fails)
- Sub-linear query time: O(n^ρ) where ρ < 1
- Simple and easy to implement

Disadvantages:
- Approximate (not guaranteed to find true nearest neighbors)
- Memory intensive (multiple tables)
- Recall-space trade-off

Time Complexity:
    - Build: O(n * L * k * d) where n=vectors, L=tables, k=bits, d=dimension
    - Search: O(L * bucket_size + k_candidates * d)

Space Complexity: O(n * L)
"""

from typing import List, Optional, Dict, Any, Set, Tuple
from collections import defaultdict
import numpy as np
import heapq

from ..core.base import BaseIndex, SearchResult
from ..core.vector import Vector, VectorOps
from ..core.distance import Distance, DistanceMetric


class LSHTable:
    """A single LSH hash table with random projection.

    Each table uses k random hyperplanes to create k-bit hash codes.
    Vectors are stored in buckets indexed by their hash codes.
    """

    def __init__(self, dimension: int, num_bits: int, seed: Optional[int] = None):
        """Initialize an LSH table.

        Args:
            dimension: Dimensionality of vectors
            num_bits: Number of hash bits (number of random hyperplanes)
            seed: Random seed for reproducibility
        """
        self.dimension = dimension
        self.num_bits = num_bits

        # Generate random hyperplanes (unit vectors)
        rng = np.random.RandomState(seed)
        self.hyperplanes = []
        for _ in range(num_bits):
            # Generate random unit vector
            plane = rng.randn(dimension).astype(np.float32)
            plane = plane / np.linalg.norm(plane)
            self.hyperplanes.append(plane)

        # Hash table: hash_code -> list of indices
        self.buckets: Dict[int, List[int]] = defaultdict(list)

    def hash(self, vector: Vector) -> int:
        """Compute hash code for a vector using random projection.

        The hash is computed as:
        h(v) = [h_1(v), h_2(v), ..., h_k(v)]
        where h_i(v) = 1 if v · r_i > 0 else 0

        This creates a k-bit binary hash code.

        Args:
            vector: Input vector

        Returns:
            Integer hash code (interpreted as binary)
        """
        hash_code = 0
        for i, hyperplane in enumerate(self.hyperplanes):
            # Check which side of hyperplane the vector is on
            dot_product = np.dot(vector, hyperplane)
            if dot_product > 0:
                # Set bit i to 1
                hash_code |= (1 << i)

        return hash_code

    def add(self, vector: Vector, index: int) -> None:
        """Add a vector to the hash table.

        Args:
            vector: Vector to add
            index: Index of the vector
        """
        hash_code = self.hash(vector)
        self.buckets[hash_code].append(index)

    def query(self, vector: Vector) -> List[int]:
        """Query the hash table for candidate neighbors.

        Args:
            vector: Query vector

        Returns:
            List of candidate indices in the same bucket
        """
        hash_code = self.hash(vector)
        return self.buckets.get(hash_code, [])

    def get_bucket_sizes(self) -> List[int]:
        """Get sizes of all buckets.

        Returns:
            List of bucket sizes
        """
        return [len(bucket) for bucket in self.buckets.values()]

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"LSHTable(dimension={self.dimension}, "
            f"num_bits={self.num_bits}, "
            f"num_buckets={len(self.buckets)})"
        )


class LSH(BaseIndex):
    """Locality Sensitive Hashing index with random projection.

    This implementation uses:
    - Random projection (SimHash) for cosine similarity
    - Multi-table strategy for higher recall
    - Configurable number of tables and hash bits

    The key insight: vectors that are close in angle (cosine similarity)
    will likely have the same hash code, falling into the same bucket.
    """

    def __init__(
        self,
        dimension: int,
        num_tables: int = 5,
        num_bits: int = 10,
        metric: DistanceMetric = DistanceMetric.COSINE,
        seed: int = 42
    ):
        """Initialize the LSH index.

        Args:
            dimension: Dimensionality of vectors
            num_tables: Number of hash tables (more = higher recall, more memory)
            num_bits: Number of hash bits per table (more = fewer collisions)
            metric: Distance metric (COSINE works best with random projection)
            seed: Random seed for reproducibility

        Note:
            Random projection LSH is designed for cosine similarity.
            For L2 distance, you would need different hash functions (e.g., p-stable).
        """
        super().__init__(dimension)
        self.num_tables = num_tables
        self.num_bits = num_bits
        self.metric = metric
        self.seed = seed

        # Create L hash tables with different random projections
        self.tables: List[LSHTable] = []
        for i in range(num_tables):
            table = LSHTable(dimension, num_bits, seed=seed + i)
            self.tables.append(table)

        self._vectors: List[Vector] = []
        self._metadata: List[Optional[Dict[str, Any]]] = []
        self._distance_func = Distance.get_distance_function(metric)

        # Statistics
        self._total_candidates = 0
        self._total_queries = 0

    def add(
        self,
        vectors: List[Vector],
        metadata: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """Add vectors to the LSH index.

        Args:
            vectors: List of vectors to add
            metadata: Optional metadata for each vector

        Note:
            For cosine similarity, vectors should ideally be normalized.
        """
        self._validate_vectors(vectors)

        start_time = self._start_build_timer()

        start_idx = len(self._vectors)
        self._vectors.extend(vectors)

        # Handle metadata
        if metadata is None:
            self._metadata.extend([None] * len(vectors))
        else:
            if len(metadata) != len(vectors):
                raise ValueError("Metadata length doesn't match vectors length")
            self._metadata.extend(metadata)

        # Add vectors to all hash tables
        for i, vector in enumerate(vectors):
            vector_idx = start_idx + i
            for table in self.tables:
                table.add(vector, vector_idx)

        self.num_vectors = len(self._vectors)
        self._end_build_timer(start_time)

    def search(
        self,
        query: Vector,
        k: int = 10,
        max_candidates: Optional[int] = None,
        **kwargs
    ) -> SearchResult:
        """Search for k nearest neighbors using LSH.

        Algorithm:
        1. Hash query in all L tables
        2. Collect candidate indices from all buckets
        3. Compute exact distances to all candidates
        4. Return top-k by distance

        Args:
            query: Query vector
            k: Number of nearest neighbors to return
            max_candidates: Maximum candidates to consider (for speed)
            **kwargs: Additional parameters

        Returns:
            SearchResult with k nearest neighbors

        Note:
            More candidates = higher recall but slower search.
            max_candidates provides a recall-speed trade-off.
        """
        self._validate_dimension(query)

        if self.num_vectors == 0:
            return SearchResult(indices=[], distances=[], metadata=[])

        if k > self.num_vectors:
            k = self.num_vectors

        # Collect candidates from all hash tables
        candidates: Set[int] = set()
        for table in self.tables:
            bucket_candidates = table.query(query)
            candidates.update(bucket_candidates)

        # Track statistics
        self._total_candidates += len(candidates)
        self._total_queries += 1

        # If no candidates found, fall back to random or empty
        if not candidates:
            return SearchResult(indices=[], distances=[], metadata=[])

        # Limit candidates if specified
        if max_candidates is not None and len(candidates) > max_candidates:
            candidates = set(list(candidates)[:max_candidates])

        # Compute exact distances to all candidates
        distances = []
        for idx in candidates:
            dist = self._distance_func(query, self._vectors[idx])
            distances.append((dist, idx))

        # Get k smallest distances
        if len(distances) < k:
            k = len(distances)

        top_k = heapq.nsmallest(k, distances, key=lambda x: x[0])

        # Extract results
        result_distances = [dist for dist, _ in top_k]
        result_indices = [idx for _, idx in top_k]

        # Extract metadata
        result_metadata = None
        if any(m is not None for m in self._metadata):
            result_metadata = [self._metadata[idx] for idx in result_indices]

        return SearchResult(
            indices=result_indices,
            distances=result_distances,
            metadata=result_metadata
        )

    def get_candidate_stats(self) -> Dict[str, Any]:
        """Get statistics about candidate generation.

        Returns:
            Dictionary with candidate statistics
        """
        if self._total_queries == 0:
            return {
                "total_queries": 0,
                "avg_candidates": 0,
                "candidate_rate": 0.0
            }

        avg_candidates = self._total_candidates / self._total_queries
        candidate_rate = avg_candidates / self.num_vectors if self.num_vectors > 0 else 0

        return {
            "total_queries": self._total_queries,
            "avg_candidates": avg_candidates,
            "candidate_rate": candidate_rate,
            "candidate_percentage": candidate_rate * 100
        }

    def get_bucket_stats(self) -> Dict[str, Any]:
        """Get statistics about hash bucket distribution.

        Returns:
            Dictionary with bucket statistics
        """
        all_bucket_sizes = []
        for table in self.tables:
            all_bucket_sizes.extend(table.get_bucket_sizes())

        if not all_bucket_sizes:
            return {
                "num_tables": self.num_tables,
                "total_buckets": 0,
                "avg_bucket_size": 0,
                "max_bucket_size": 0,
                "min_bucket_size": 0
            }

        return {
            "num_tables": self.num_tables,
            "total_buckets": len(all_bucket_sizes),
            "avg_bucket_size": np.mean(all_bucket_sizes),
            "max_bucket_size": np.max(all_bucket_sizes),
            "min_bucket_size": np.min(all_bucket_sizes),
            "std_bucket_size": np.std(all_bucket_sizes)
        }

    def _estimate_memory(self) -> int:
        """Estimate memory usage in bytes.

        Returns:
            Approximate memory usage
        """
        # Vectors storage
        vectors_mem = self.num_vectors * self.dimension * 4

        # Hash tables (indices are int64 = 8 bytes)
        # Each vector appears in num_tables buckets
        tables_mem = self.num_vectors * self.num_tables * 8

        # Random hyperplanes
        hyperplanes_mem = self.num_tables * self.num_bits * self.dimension * 4

        return vectors_mem + tables_mem + hyperplanes_mem

    def _get_additional_stats(self) -> Dict[str, Any]:
        """Get LSH specific statistics."""
        stats = {
            "metric": self.metric.value,
            "num_tables": self.num_tables,
            "num_bits": self.num_bits,
            "hash_functions": self.num_tables * self.num_bits,
        }

        # Add candidate stats
        stats.update(self.get_candidate_stats())

        # Add bucket stats
        stats.update(self.get_bucket_stats())

        return stats

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"LSH(dimension={self.dimension}, "
            f"num_vectors={self.num_vectors}, "
            f"num_tables={self.num_tables}, "
            f"num_bits={self.num_bits}, "
            f"metric={self.metric.value})"
        )
