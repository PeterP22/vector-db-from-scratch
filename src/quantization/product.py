"""Product Quantization (PQ) for extreme vector compression.

Product Quantization achieves 16-32x compression by splitting vectors into
subspaces and quantizing each subspace independently using k-means clustering.

Key Concepts:
1. Subspace splitting: Split d-dimensional vector into m subspaces of d/m dimensions
2. Codebook: K-means cluster each subspace independently → m codebooks
3. Encoding: Replace each subspace with its cluster ID (1 byte)
4. Asymmetric Distance Computation (ADC): Query stays full precision, DB is quantized

Algorithm:
- Train: K-means on each subspace → m codebooks of k centroids
- Encode: vector → [cluster_id_1, cluster_id_2, ..., cluster_id_m]
- Decode: [id_1, ..., id_m] → concatenate centroids from codebooks
- ADC: Precompute query distances to all centroids, lookup during search

Example (d=128, m=8, k=256):
- Original: 128 dims × 4 bytes = 512 bytes
- PQ: 8 subspaces × 1 byte = 8 bytes
- Compression: 64x!

Benefits:
- 16-64x compression (vs 4x for scalar quantization)
- Asymmetric distance is very accurate
- Standard in FAISS (IVF+PQ)

Trade-offs:
- Training is expensive (m × k-means)
- Encoding/decoding slower than scalar
- Requires careful tuning of m and k

Production Usage:
    - FAISS: IVF+PQ for billion-scale search
    - Milvus: PQ support for compression
    - Often combined with IVF for coarse-to-fine search
"""

from typing import List, Optional, Tuple
import numpy as np
from dataclasses import dataclass

from ..core.vector import Vector


@dataclass
class ProductQuantizerParams:
    """Parameters for product quantization.

    Attributes:
        num_subspaces: Number of subspaces (m)
        num_clusters: Number of clusters per subspace (k)
        subspace_dim: Dimension of each subspace (d/m)
        codebooks: List of codebook matrices, each of shape (k, subspace_dim)
    """

    num_subspaces: int
    num_clusters: int
    subspace_dim: int
    codebooks: List[np.ndarray]


class ProductQuantizer:
    """Product Quantization for extreme vector compression.

    PQ splits vectors into subspaces and quantizes each independently,
    achieving much higher compression than scalar quantization.
    """

    def __init__(
        self,
        dimension: int,
        num_subspaces: int = 8,
        num_clusters: int = 256,
        kmeans_iterations: int = 20,
        seed: int = 42
    ):
        """Initialize product quantizer.

        Args:
            dimension: Dimensionality of vectors
            num_subspaces: Number of subspaces (m). Must divide dimension evenly.
            num_clusters: Number of clusters per subspace (k). Usually 256 (1 byte).
            kmeans_iterations: K-means iterations for training
            seed: Random seed

        Raises:
            ValueError: If dimension is not divisible by num_subspaces
        """
        if dimension % num_subspaces != 0:
            raise ValueError(
                f"Dimension {dimension} must be divisible by num_subspaces {num_subspaces}"
            )

        self.dimension = dimension
        self.num_subspaces = num_subspaces
        self.num_clusters = num_clusters
        self.kmeans_iterations = kmeans_iterations
        self.subspace_dim = dimension // num_subspaces
        self.seed = seed

        self.params: Optional[ProductQuantizerParams] = None
        self._is_trained = False

        self._rng = np.random.RandomState(seed)

    def _kmeans_subspace(
        self,
        subspace_data: np.ndarray,
        k: int
    ) -> np.ndarray:
        """Run k-means on a subspace.

        Args:
            subspace_data: Data matrix of shape (n, subspace_dim)
            k: Number of clusters

        Returns:
            Centroids matrix of shape (k, subspace_dim)
        """
        n_samples = subspace_data.shape[0]

        # Initialize centroids randomly (with replacement if k > n_samples)
        replace = k > n_samples
        centroid_indices = self._rng.choice(n_samples, k, replace=replace)
        centroids = subspace_data[centroid_indices].copy()

        # Lloyd's algorithm
        for _ in range(self.kmeans_iterations):
            # Assign to nearest centroid
            distances = np.linalg.norm(
                subspace_data[:, np.newaxis] - centroids,
                axis=2
            )
            labels = np.argmin(distances, axis=1)

            # Update centroids
            for i in range(k):
                cluster_points = subspace_data[labels == i]
                if len(cluster_points) > 0:
                    centroids[i] = cluster_points.mean(axis=0)

        return centroids

    def train(self, vectors: List[Vector]) -> None:
        """Train product quantizer by running k-means on each subspace.

        Args:
            vectors: Training vectors

        Note:
            This runs m independent k-means, which can be slow.
            For large datasets, consider sampling.
        """
        if not vectors:
            raise ValueError("Need training vectors")

        # Stack into matrix
        X = np.vstack(vectors)

        # Split into subspaces and train k-means on each
        codebooks = []

        for m in range(self.num_subspaces):
            # Extract subspace
            start_idx = m * self.subspace_dim
            end_idx = start_idx + self.subspace_dim
            subspace_data = X[:, start_idx:end_idx]

            # Run k-means
            centroids = self._kmeans_subspace(subspace_data, self.num_clusters)
            codebooks.append(centroids)

        self.params = ProductQuantizerParams(
            num_subspaces=self.num_subspaces,
            num_clusters=self.num_clusters,
            subspace_dim=self.subspace_dim,
            codebooks=codebooks
        )
        self._is_trained = True

    def encode(self, vector: Vector) -> np.ndarray:
        """Encode vector to PQ codes.

        Args:
            vector: Vector to encode

        Returns:
            Array of cluster IDs, shape (num_subspaces,), dtype uint8 or uint16

        Note:
            If num_clusters <= 256, uses uint8 (1 byte per subspace)
            Otherwise uses uint16 (2 bytes per subspace)
        """
        if not self._is_trained:
            raise ValueError("Must train before encoding")

        codes = np.zeros(self.num_subspaces, dtype=np.uint16 if self.num_clusters > 256 else np.uint8)

        for m in range(self.num_subspaces):
            # Extract subspace
            start_idx = m * self.subspace_dim
            end_idx = start_idx + self.subspace_dim
            subspace_vector = vector[start_idx:end_idx]

            # Find nearest centroid
            codebook = self.params.codebooks[m]
            distances = np.linalg.norm(codebook - subspace_vector, axis=1)
            codes[m] = np.argmin(distances)

        return codes

    def decode(self, codes: np.ndarray) -> Vector:
        """Decode PQ codes back to vector.

        Args:
            codes: Array of cluster IDs, shape (num_subspaces,)

        Returns:
            Reconstructed vector

        Note:
            Reconstruction is lossy - returns centroid concatenation.
        """
        if not self._is_trained:
            raise ValueError("Must train before decoding")

        reconstructed = np.zeros(self.dimension, dtype=np.float32)

        for m in range(self.num_subspaces):
            start_idx = m * self.subspace_dim
            end_idx = start_idx + self.subspace_dim

            # Get centroid for this subspace
            centroid = self.params.codebooks[m][codes[m]]
            reconstructed[start_idx:end_idx] = centroid

        return reconstructed

    def encode_batch(self, vectors: List[Vector]) -> np.ndarray:
        """Encode multiple vectors efficiently.

        Args:
            vectors: List of vectors

        Returns:
            Matrix of codes, shape (n_vectors, num_subspaces)
        """
        return np.vstack([self.encode(v) for v in vectors])

    def decode_batch(self, codes: np.ndarray) -> List[Vector]:
        """Decode multiple PQ codes efficiently.

        Args:
            codes: Matrix of codes, shape (n_vectors, num_subspaces)

        Returns:
            List of reconstructed vectors
        """
        return [self.decode(codes[i]) for i in range(len(codes))]

    def compute_distance_table(self, query: Vector) -> np.ndarray:
        """Compute distance table for Asymmetric Distance Computation (ADC).

        This precomputes distances from query subspaces to all centroids.
        Used for fast search with ADC.

        Args:
            query: Query vector (full precision)

        Returns:
            Distance table of shape (num_subspaces, num_clusters)
            table[m, k] = distance from query subspace m to centroid k

        Note:
            This is the key to efficient PQ search!
        """
        if not self._is_trained:
            raise ValueError("Must train before computing distance table")

        distance_table = np.zeros((self.num_subspaces, self.num_clusters), dtype=np.float32)

        for m in range(self.num_subspaces):
            # Extract query subspace
            start_idx = m * self.subspace_dim
            end_idx = start_idx + self.subspace_dim
            query_subspace = query[start_idx:end_idx]

            # Compute distances to all centroids in this subspace
            codebook = self.params.codebooks[m]
            distances = np.linalg.norm(codebook - query_subspace, axis=1)
            distance_table[m] = distances

        return distance_table

    def adc_distance(
        self,
        codes: np.ndarray,
        distance_table: np.ndarray
    ) -> float:
        """Compute Asymmetric Distance using precomputed distance table.

        Formula: distance = sqrt(sum(distance_table[m, codes[m]]^2))

        Args:
            codes: PQ codes for database vector
            distance_table: Precomputed distance table from query

        Returns:
            Approximate L2 distance

        Note:
            This is MUCH faster than decoding and computing full distance.
        """
        # Lookup distances from table
        squared_dist = 0.0
        for m in range(self.num_subspaces):
            squared_dist += distance_table[m, codes[m]] ** 2

        return np.sqrt(squared_dist)

    def adc_distance_batch(
        self,
        codes_batch: np.ndarray,
        distance_table: np.ndarray
    ) -> np.ndarray:
        """Compute ADC distances for multiple database vectors.

        Args:
            codes_batch: Matrix of codes, shape (n_vectors, num_subspaces)
            distance_table: Precomputed distance table

        Returns:
            Array of distances, shape (n_vectors,)
        """
        # Vectorized lookup
        squared_dists = np.zeros(len(codes_batch), dtype=np.float32)

        for m in range(self.num_subspaces):
            # Lookup distances for all vectors at once
            dists_m = distance_table[m, codes_batch[:, m]]
            squared_dists += dists_m ** 2

        return np.sqrt(squared_dists)

    def get_compression_ratio(self) -> float:
        """Get compression ratio achieved by PQ.

        Returns:
            Compression ratio
        """
        bytes_per_code = 1 if self.num_clusters <= 256 else 2
        original_bytes = self.dimension * 4  # float32
        compressed_bytes = self.num_subspaces * bytes_per_code

        return original_bytes / compressed_bytes

    def get_memory_overhead(self) -> int:
        """Get memory overhead for storing codebooks.

        Returns:
            Memory overhead in bytes
        """
        # m codebooks of shape (k, subspace_dim)
        return self.num_subspaces * self.num_clusters * self.subspace_dim * 4

    def __repr__(self) -> str:
        """String representation."""
        trained_str = "trained" if self._is_trained else "untrained"
        return (
            f"ProductQuantizer(dimension={self.dimension}, "
            f"num_subspaces={self.num_subspaces}, "
            f"num_clusters={self.num_clusters}, "
            f"{trained_str})"
        )


def compute_pq_error(
    original: List[Vector],
    quantizer: ProductQuantizer
) -> Tuple[float, float, float]:
    """Compute PQ reconstruction error.

    Args:
        original: Original vectors
        quantizer: Trained PQ quantizer

    Returns:
        Tuple of (mean_error, max_error, relative_error)
    """
    errors = []

    for vector in original:
        codes = quantizer.encode(vector)
        reconstructed = quantizer.decode(codes)
        error = np.linalg.norm(vector - reconstructed)
        errors.append(error)

    mean_error = np.mean(errors)
    max_error = np.max(errors)

    magnitudes = [np.linalg.norm(v) for v in original]
    relative_error = mean_error / np.mean(magnitudes) if np.mean(magnitudes) > 0 else 0

    return mean_error, max_error, relative_error


if __name__ == "__main__":
    # Quick demo
    print("Product Quantization Demo")
    print("=" * 50)

    dimension = 128
    n_vectors = 1000

    # Generate random vectors
    vectors = [np.random.randn(dimension).astype(np.float32) for _ in range(n_vectors)]

    # Train PQ
    pq = ProductQuantizer(dimension, num_subspaces=8, num_clusters=256)
    print(f"\nTraining {pq}...")
    pq.train(vectors)

    # Compute compression
    print(f"\nCompression:")
    print(f"  Original: {dimension * 4} bytes per vector")
    print(f"  PQ: {pq.num_subspaces} bytes per vector")
    print(f"  Ratio: {pq.get_compression_ratio():.1f}x")
    print(f"  Codebook overhead: {pq.get_memory_overhead() / 1024:.2f} KB")

    # Compute error
    mean_err, max_err, rel_err = compute_pq_error(vectors[:100], pq)
    print(f"\nReconstruction Error:")
    print(f"  Mean: {mean_err:.4f}")
    print(f"  Max: {max_err:.4f}")
    print(f"  Relative: {rel_err:.2%}")

    # ADC demo
    query = vectors[0]
    target = vectors[1]

    # Compute using ADC
    distance_table = pq.compute_distance_table(query)
    target_codes = pq.encode(target)
    adc_dist = pq.adc_distance(target_codes, distance_table)

    # Compare to true distance
    true_dist = float(np.linalg.norm(query - target))

    print(f"\nADC Distance Test:")
    print(f"  True distance: {true_dist:.4f}")
    print(f"  ADC distance: {adc_dist:.4f}")
    print(f"  Relative error: {abs(true_dist - adc_dist) / true_dist:.2%}")
