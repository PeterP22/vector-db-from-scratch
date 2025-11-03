"""Generate test datasets for vector database evaluation."""

from typing import List, Tuple
from enum import Enum
import numpy as np

from ..core.vector import Vector, VectorOps


class DatasetType(Enum):
    """Types of synthetic datasets for testing."""

    RANDOM = "random"  # Uniformly random vectors
    GAUSSIAN = "gaussian"  # Gaussian distributed vectors
    CLUSTERED = "clustered"  # Vectors grouped in clusters
    NORMALIZED = "normalized"  # Unit-length vectors (for cosine similarity)


class DataGenerator:
    """Generate synthetic vector datasets for testing and benchmarking.

    This class provides various methods to generate test data with different
    characteristics to evaluate vector database performance.
    """

    def __init__(self, seed: int = 42):
        """Initialize the data generator.

        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        np.random.seed(seed)

    def generate(
        self,
        num_vectors: int,
        dimension: int,
        dataset_type: DatasetType = DatasetType.RANDOM,
        **kwargs
    ) -> List[Vector]:
        """Generate a synthetic dataset.

        Args:
            num_vectors: Number of vectors to generate
            dimension: Dimensionality of vectors
            dataset_type: Type of dataset to generate
            **kwargs: Additional parameters specific to dataset type

        Returns:
            List of generated vectors
        """
        if dataset_type == DatasetType.RANDOM:
            return self._generate_random(num_vectors, dimension)
        elif dataset_type == DatasetType.GAUSSIAN:
            return self._generate_gaussian(num_vectors, dimension, **kwargs)
        elif dataset_type == DatasetType.CLUSTERED:
            return self._generate_clustered(num_vectors, dimension, **kwargs)
        elif dataset_type == DatasetType.NORMALIZED:
            return self._generate_normalized(num_vectors, dimension)
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")

    def _generate_random(self, num_vectors: int, dimension: int) -> List[Vector]:
        """Generate uniformly random vectors in [0, 1].

        Args:
            num_vectors: Number of vectors
            dimension: Vector dimension

        Returns:
            List of random vectors
        """
        return [
            np.random.rand(dimension).astype(np.float32)
            for _ in range(num_vectors)
        ]

    def _generate_gaussian(
        self,
        num_vectors: int,
        dimension: int,
        mean: float = 0.0,
        std: float = 1.0
    ) -> List[Vector]:
        """Generate vectors from a Gaussian distribution.

        Args:
            num_vectors: Number of vectors
            dimension: Vector dimension
            mean: Mean of the Gaussian distribution
            std: Standard deviation of the Gaussian distribution

        Returns:
            List of Gaussian-distributed vectors
        """
        return [
            np.random.normal(mean, std, dimension).astype(np.float32)
            for _ in range(num_vectors)
        ]

    def _generate_clustered(
        self,
        num_vectors: int,
        dimension: int,
        num_clusters: int = 10,
        cluster_std: float = 0.1
    ) -> List[Vector]:
        """Generate clustered vectors (Gaussian mixture).

        Args:
            num_vectors: Number of vectors
            dimension: Vector dimension
            num_clusters: Number of clusters
            cluster_std: Standard deviation within each cluster

        Returns:
            List of clustered vectors
        """
        # Generate cluster centroids
        centroids = [
            np.random.randn(dimension).astype(np.float32)
            for _ in range(num_clusters)
        ]

        vectors = []
        vectors_per_cluster = num_vectors // num_clusters
        remainder = num_vectors % num_clusters

        for i, centroid in enumerate(centroids):
            # Add extra vectors to first clusters if there's a remainder
            count = vectors_per_cluster + (1 if i < remainder else 0)

            for _ in range(count):
                # Add Gaussian noise around centroid
                noise = np.random.normal(0, cluster_std, dimension).astype(np.float32)
                vectors.append(centroid + noise)

        # Shuffle to mix clusters
        np.random.shuffle(vectors)
        return vectors

    def _generate_normalized(self, num_vectors: int, dimension: int) -> List[Vector]:
        """Generate unit-length vectors (normalized to L2 norm = 1).

        Args:
            num_vectors: Number of vectors
            dimension: Vector dimension

        Returns:
            List of normalized vectors
        """
        vectors = self._generate_gaussian(num_vectors, dimension)
        return VectorOps.batch_normalize(vectors)

    def generate_query_vectors(
        self,
        dataset: List[Vector],
        num_queries: int,
        noise_level: float = 0.1
    ) -> List[Vector]:
        """Generate query vectors by adding noise to dataset vectors.

        This creates queries that should have known nearest neighbors,
        useful for evaluating recall.

        Args:
            dataset: Original dataset
            num_queries: Number of query vectors to generate
            noise_level: Standard deviation of Gaussian noise to add

        Returns:
            List of query vectors
        """
        if num_queries > len(dataset):
            num_queries = len(dataset)

        # Sample random vectors from dataset
        indices = np.random.choice(len(dataset), num_queries, replace=False)

        queries = []
        for idx in indices:
            base_vector = dataset[idx]
            # Add Gaussian noise
            noise = np.random.normal(0, noise_level, len(base_vector)).astype(np.float32)
            query = base_vector + noise
            queries.append(query)

        return queries

    def generate_dataset_with_queries(
        self,
        num_vectors: int,
        dimension: int,
        num_queries: int = 100,
        dataset_type: DatasetType = DatasetType.RANDOM,
        **kwargs
    ) -> Tuple[List[Vector], List[Vector]]:
        """Generate both a dataset and corresponding query vectors.

        Args:
            num_vectors: Number of vectors in dataset
            dimension: Vector dimension
            num_queries: Number of query vectors
            dataset_type: Type of dataset
            **kwargs: Additional parameters for dataset generation

        Returns:
            Tuple of (dataset_vectors, query_vectors)
        """
        dataset = self.generate(num_vectors, dimension, dataset_type, **kwargs)
        queries = self.generate_query_vectors(dataset, num_queries)
        return dataset, queries

    def load_synthetic_benchmark(
        self,
        size: str = "small"
    ) -> Tuple[List[Vector], List[Vector], int]:
        """Load a pre-configured synthetic benchmark dataset.

        Args:
            size: Benchmark size ("small", "medium", "large")

        Returns:
            Tuple of (dataset_vectors, query_vectors, dimension)
        """
        configs = {
            "small": {"num_vectors": 1000, "dimension": 128, "num_queries": 100},
            "medium": {"num_vectors": 10000, "dimension": 256, "num_queries": 500},
            "large": {"num_vectors": 100000, "dimension": 512, "num_queries": 1000},
        }

        if size not in configs:
            raise ValueError(f"Unknown size: {size}. Choose from {list(configs.keys())}")

        config = configs[size]
        dataset, queries = self.generate_dataset_with_queries(
            config["num_vectors"],
            config["dimension"],
            config["num_queries"],
            DatasetType.CLUSTERED,
            num_clusters=50
        )

        return dataset, queries, config["dimension"]

    def print_dataset_info(self, dataset: List[Vector], name: str = "Dataset") -> None:
        """Print information about a dataset.

        Args:
            dataset: List of vectors
            name: Name for display
        """
        if not dataset:
            print(f"{name}: Empty")
            return

        dimension = len(dataset[0])
        count = len(dataset)

        print(f"\n{name} Information:")
        print(f"  Vectors: {count:,}")
        print(f"  Dimensions: {dimension}")
        print(f"  Memory (raw): {count * dimension * 4 / 1024 / 1024:.2f} MB")

        # Sample statistics
        sample_size = min(100, count)
        sample = np.random.choice(count, sample_size, replace=False)
        sample_vectors = [dataset[i] for i in sample]

        magnitudes = [VectorOps.magnitude(v) for v in sample_vectors]
        print(f"  Magnitude (sample) - Mean: {np.mean(magnitudes):.4f}, "
              f"Std: {np.std(magnitudes):.4f}")
