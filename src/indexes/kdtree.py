"""KD-Tree index for nearest neighbor search.

A KD-Tree (k-dimensional tree) is a space-partitioning data structure that
recursively divides the space using axis-aligned hyperplanes. At each level,
the tree splits along a different dimension (cycling through dimensions).

Key Concepts:
- Binary Space Partitioning (BSP) tree
- Split dimension cycles with depth: split_dim = depth % d
- Hyperplane perpendicular to axis at depth % d
- Bound and branch pruning with greedy DFS
- Works well in low dimensions (d < 20), degrades with high dimensions (curse of dimensionality)

Time Complexity:
    - Build: O(n log² n)
    - Search (ideal): O(log n)
    - Search (worst): O(n) - in high dimensions or unbalanced trees

Space Complexity: O(n)
"""

from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass
import heapq
import numpy as np

from ..core.base import BaseIndex, SearchResult
from ..core.vector import Vector
from ..core.distance import Distance, DistanceMetric


@dataclass
class KDNode:
    """Node in the KD-Tree.

    Attributes:
        vector: The vector stored at this node
        index: Original index of the vector in the dataset
        split_dim: Dimension used for splitting at this node
        left: Left child (values <= split value)
        right: Right child (values > split value)
        metadata: Optional metadata for this vector
    """

    vector: Vector
    index: int
    split_dim: int
    left: Optional['KDNode'] = None
    right: Optional['KDNode'] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class SearchStats:
    """Statistics tracked during search operations.

    Used to analyze pruning effectiveness and optimize the tree.
    """

    nodes_visited: int = 0
    nodes_pruned: int = 0
    distance_computations: int = 0


class KDTree(BaseIndex):
    """KD-Tree index for nearest neighbor search.

    This implementation includes:
    - Recursive BST insertion with dimension cycling (depth % d)
    - Hyperplane perpendicular splitting to axis
    - Bound and branch pruning with greedy DFS
    - Statistics tracking for optimization analysis

    Note:
        KD-Trees work best for low-dimensional data (d < 20).
        Performance degrades significantly in high dimensions due to
        the curse of dimensionality - most of the space is far from
        the query, so pruning becomes less effective.
    """

    def __init__(
        self,
        dimension: int,
        metric: DistanceMetric = DistanceMetric.L2,
        leaf_size: int = 1
    ):
        """Initialize the KD-Tree index.

        Args:
            dimension: Dimensionality of vectors
            metric: Distance metric to use (only L2 fully supported)
            leaf_size: Minimum number of points in a leaf (not yet implemented)
        """
        super().__init__(dimension)
        self.metric = metric
        self.leaf_size = leaf_size
        self.root: Optional[KDNode] = None
        self._distance_func = Distance.get_distance_function(metric)
        self._vectors: List[Vector] = []
        self._metadata: List[Optional[Dict[str, Any]]] = []

        # Statistics for optimization analysis
        self._total_nodes_visited = 0
        self._total_nodes_pruned = 0
        self._total_searches = 0

    def add(
        self,
        vectors: List[Vector],
        metadata: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """Add vectors to the KD-Tree.

        Args:
            vectors: List of vectors to add
            metadata: Optional metadata for each vector

        Note:
            This rebuilds the entire tree. For incremental insertion,
            use a different data structure or rebuild periodically.
        """
        self._validate_vectors(vectors)

        start_time = self._start_build_timer()

        # Store vectors and metadata
        start_idx = len(self._vectors)
        self._vectors.extend(vectors)

        if metadata is None:
            self._metadata.extend([None] * len(vectors))
        else:
            if len(metadata) != len(vectors):
                raise ValueError("Metadata length doesn't match vectors length")
            self._metadata.extend(metadata)

        # Create indices for new vectors
        indices = list(range(start_idx, start_idx + len(vectors)))

        # Build tree from all vectors
        self.root = self._build_tree(indices, depth=0)
        self.num_vectors = len(self._vectors)

        self._end_build_timer(start_time)

    def _build_tree(
        self,
        indices: List[int],
        depth: int
    ) -> Optional[KDNode]:
        """Recursively build the KD-Tree using BST insertion.

        This implements:
        - Dimension cycling: split_dim = depth % d
        - Median-based partitioning for balanced tree
        - Hyperplane perpendicular to axis at split_dim

        Args:
            indices: Indices of vectors to include in this subtree
            depth: Current depth in the tree

        Returns:
            Root node of the subtree, or None if no vectors
        """
        if not indices:
            return None

        if len(indices) == 1:
            # Leaf node
            idx = indices[0]
            return KDNode(
                vector=self._vectors[idx],
                index=idx,
                split_dim=depth % self.dimension,
                metadata=self._metadata[idx]
            )

        # Cycle through dimensions
        split_dim = depth % self.dimension

        # Sort indices by the split dimension and find median
        # This creates a balanced tree
        sorted_indices = sorted(
            indices,
            key=lambda idx: self._vectors[idx][split_dim]
        )
        median_idx = len(sorted_indices) // 2

        # Create node at median
        node_idx = sorted_indices[median_idx]
        node = KDNode(
            vector=self._vectors[node_idx],
            index=node_idx,
            split_dim=split_dim,
            metadata=self._metadata[node_idx]
        )

        # Recursively build left and right subtrees
        # Left: all points with split_dim <= median
        # Right: all points with split_dim > median
        node.left = self._build_tree(sorted_indices[:median_idx], depth + 1)
        node.right = self._build_tree(sorted_indices[median_idx + 1:], depth + 1)

        return node

    def search(
        self,
        query: Vector,
        k: int = 10,
        track_stats: bool = False
    ) -> SearchResult:
        """Search for k nearest neighbors using bound and branch pruning.

        This implements greedy DFS search with pruning:
        1. Start at root, maintain best-k candidates
        2. At each node, compute distance and update candidates
        3. Recursively search subtrees in order (greedy: closest first)
        4. Prune subtrees that can't contain better candidates

        Args:
            query: Query vector
            k: Number of nearest neighbors to return
            track_stats: Whether to track search statistics

        Returns:
            SearchResult with k nearest neighbors

        Raises:
            ValueError: If query dimension doesn't match
        """
        self._validate_dimension(query)

        if self.root is None or self.num_vectors == 0:
            return SearchResult(indices=[], distances=[], metadata=[])

        if k > self.num_vectors:
            k = self.num_vectors

        # Max heap to track k best candidates (using negative distances)
        # heapq is a min heap, so we negate distances to get max heap behavior
        best_k: List[Tuple[float, int]] = []  # [(negative_distance, index), ...]

        # Track statistics if requested
        stats = SearchStats() if track_stats else None

        # Perform DFS search with pruning
        self._search_recursive(query, self.root, k, best_k, stats)

        # Update global statistics
        if stats:
            self._total_nodes_visited += stats.nodes_visited
            self._total_nodes_pruned += stats.nodes_pruned
            self._total_searches += 1

        # Extract results (convert back from negative distances)
        best_k.sort(reverse=True)  # Sort by distance (closest first)
        result_distances = [-dist for dist, _ in best_k]
        result_indices = [idx for _, idx in best_k]

        # Extract metadata
        result_metadata = None
        if any(m is not None for m in self._metadata):
            result_metadata = [self._metadata[idx] for idx in result_indices]

        return SearchResult(
            indices=result_indices,
            distances=result_distances,
            metadata=result_metadata
        )

    def _search_recursive(
        self,
        query: Vector,
        node: Optional[KDNode],
        k: int,
        best_k: List[Tuple[float, int]],
        stats: Optional[SearchStats]
    ) -> None:
        """Recursively search the tree with bound and branch pruning.

        Key optimization: Greedy DFS
        - Visit the subtree that's likely closer first
        - This helps us find good candidates early, enabling more pruning

        Args:
            query: Query vector
            node: Current node to search
            k: Number of neighbors to find
            best_k: Heap of current best k candidates (negative distances)
            stats: Statistics tracker (if enabled)
        """
        if node is None:
            return

        if stats:
            stats.nodes_visited += 1

        # Compute distance to current node
        dist = self._distance_func(query, node.vector)

        if stats:
            stats.distance_computations += 1

        # Update best-k candidates
        if len(best_k) < k:
            heapq.heappush(best_k, (-dist, node.index))
        elif dist < -best_k[0][0]:  # better than worst in best_k
            heapq.heapreplace(best_k, (-dist, node.index))

        # Determine which subtree to search first (greedy)
        # If query[split_dim] <= node.vector[split_dim], go left first
        split_dim = node.split_dim
        diff = query[split_dim] - node.vector[split_dim]

        if diff <= 0:
            # Query is on left side, search left first (greedy)
            near_subtree = node.left
            far_subtree = node.right
        else:
            # Query is on right side, search right first (greedy)
            near_subtree = node.right
            far_subtree = node.left

        # Always search near subtree
        self._search_recursive(query, near_subtree, k, best_k, stats)

        # Pruning decision: Should we search the far subtree?
        # Only search if the hyperplane is within the current k-th best distance
        if len(best_k) < k:
            # Haven't found k candidates yet, must search
            self._search_recursive(query, far_subtree, k, best_k, stats)
        else:
            # Check if far subtree could contain better candidates
            # The closest point in far subtree is at least |diff| away
            worst_dist = -best_k[0][0]  # Current k-th best distance

            if abs(diff) < worst_dist:
                # Far subtree might contain better candidates
                self._search_recursive(query, far_subtree, k, best_k, stats)
            else:
                # Prune: far subtree can't improve results
                if stats:
                    stats.nodes_pruned += self._count_nodes(far_subtree)

    def _count_nodes(self, node: Optional[KDNode]) -> int:
        """Count nodes in a subtree (for pruning statistics).

        Args:
            node: Root of subtree

        Returns:
            Number of nodes in subtree
        """
        if node is None:
            return 0
        return 1 + self._count_nodes(node.left) + self._count_nodes(node.right)

    def get_pruning_stats(self) -> Dict[str, Any]:
        """Get statistics about pruning effectiveness.

        Returns:
            Dictionary with pruning statistics
        """
        if self._total_searches == 0:
            return {
                "total_searches": 0,
                "avg_nodes_visited": 0,
                "avg_nodes_pruned": 0,
                "avg_pruning_rate": 0.0,
                "total_nodes": self.num_vectors
            }

        avg_visited = self._total_nodes_visited / self._total_searches
        avg_pruned = self._total_nodes_pruned / self._total_searches
        pruning_rate = avg_pruned / (avg_visited + avg_pruned) if (avg_visited + avg_pruned) > 0 else 0

        return {
            "total_searches": self._total_searches,
            "avg_nodes_visited": avg_visited,
            "avg_nodes_pruned": avg_pruned,
            "avg_pruning_rate": pruning_rate,
            "pruning_percentage": pruning_rate * 100,
            "total_nodes": self.num_vectors
        }

    def _get_additional_stats(self) -> Dict[str, Any]:
        """Get KD-Tree specific statistics."""
        stats = {
            "metric": self.metric.value,
            "tree_height": self._get_height(self.root),
            "leaf_size": self.leaf_size,
        }

        # Add pruning stats if available
        if self._total_searches > 0:
            stats.update(self.get_pruning_stats())

        return stats

    def _get_height(self, node: Optional[KDNode]) -> int:
        """Calculate height of the tree.

        Args:
            node: Root node

        Returns:
            Height of tree
        """
        if node is None:
            return 0
        return 1 + max(self._get_height(node.left), self._get_height(node.right))

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"KDTree(dimension={self.dimension}, "
            f"num_vectors={self.num_vectors}, "
            f"metric={self.metric.value}, "
            f"height={self._get_height(self.root)})"
        )
