"""Base interface for all vector index implementations."""

from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Any, Dict
from dataclasses import dataclass
import time

from .vector import Vector


@dataclass
class SearchResult:
    """Result from a nearest neighbor search.

    Attributes:
        indices: Indices of the nearest neighbors
        distances: Distances to the nearest neighbors
        metadata: Optional metadata for each result
    """

    indices: List[int]
    distances: List[float]
    metadata: Optional[List[Dict[str, Any]]] = None


@dataclass
class IndexStats:
    """Statistics about an index.

    Attributes:
        num_vectors: Number of vectors in the index
        dimension: Dimensionality of vectors
        index_type: Type of index (e.g., "LinearScan", "KDTree", "HNSW")
        memory_bytes: Approximate memory usage in bytes
        build_time_seconds: Time taken to build the index
        additional_stats: Index-specific statistics
    """

    num_vectors: int
    dimension: int
    index_type: str
    memory_bytes: int
    build_time_seconds: float
    additional_stats: Dict[str, Any]


class BaseIndex(ABC):
    """Abstract base class for all vector index implementations.

    This defines the common interface that all index types must implement,
    ensuring consistent usage across different algorithms.
    """

    def __init__(self, dimension: int):
        """Initialize the index.

        Args:
            dimension: Dimensionality of vectors to be indexed
        """
        self.dimension = dimension
        self.num_vectors = 0
        self._build_time = 0.0

    @abstractmethod
    def add(self, vectors: List[Vector], metadata: Optional[List[Dict[str, Any]]] = None) -> None:
        """Add vectors to the index.

        Args:
            vectors: List of vectors to add
            metadata: Optional metadata for each vector

        Raises:
            ValueError: If vector dimensions don't match index dimension
        """
        pass

    @abstractmethod
    def search(
        self,
        query: Vector,
        k: int = 10,
        **kwargs
    ) -> SearchResult:
        """Search for k nearest neighbors of the query vector.

        Args:
            query: Query vector
            k: Number of nearest neighbors to return
            **kwargs: Index-specific search parameters

        Returns:
            SearchResult containing indices and distances of nearest neighbors

        Raises:
            ValueError: If query dimension doesn't match index dimension
        """
        pass

    def build(self) -> None:
        """Build the index (if needed).

        Some indexes require an explicit build step after adding vectors.
        Default implementation does nothing.
        """
        pass

    def get_stats(self) -> IndexStats:
        """Get statistics about the index.

        Returns:
            IndexStats object containing index information
        """
        return IndexStats(
            num_vectors=self.num_vectors,
            dimension=self.dimension,
            index_type=self.__class__.__name__,
            memory_bytes=self._estimate_memory(),
            build_time_seconds=self._build_time,
            additional_stats=self._get_additional_stats()
        )

    def _estimate_memory(self) -> int:
        """Estimate memory usage in bytes.

        Returns:
            Approximate memory usage

        Note:
            Override this in subclasses for more accurate estimates.
        """
        # Base estimate: num_vectors * dimension * 4 bytes (float32)
        return self.num_vectors * self.dimension * 4

    def _get_additional_stats(self) -> Dict[str, Any]:
        """Get index-specific statistics.

        Returns:
            Dictionary of additional statistics

        Note:
            Override this in subclasses to provide specific metrics.
        """
        return {}

    def _validate_dimension(self, vector: Vector) -> None:
        """Validate that vector has correct dimension.

        Args:
            vector: Vector to validate

        Raises:
            ValueError: If dimension doesn't match
        """
        if len(vector) != self.dimension:
            raise ValueError(
                f"Vector dimension {len(vector)} doesn't match index dimension {self.dimension}"
            )

    def _validate_vectors(self, vectors: List[Vector]) -> None:
        """Validate that all vectors have correct dimension.

        Args:
            vectors: List of vectors to validate

        Raises:
            ValueError: If any vector dimension doesn't match
        """
        for i, vector in enumerate(vectors):
            if len(vector) != self.dimension:
                raise ValueError(
                    f"Vector {i} has dimension {len(vector)}, expected {self.dimension}"
                )

    def _start_build_timer(self) -> float:
        """Start timing the build process.

        Returns:
            Start time
        """
        return time.time()

    def _end_build_timer(self, start_time: float) -> None:
        """End timing the build process and store the duration.

        Args:
            start_time: Start time from _start_build_timer
        """
        self._build_time = time.time() - start_time

    def __repr__(self) -> str:
        """String representation of the index."""
        return (
            f"{self.__class__.__name__}("
            f"dimension={self.dimension}, "
            f"num_vectors={self.num_vectors})"
        )
