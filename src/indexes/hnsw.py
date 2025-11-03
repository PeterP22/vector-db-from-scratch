"""Hierarchical Navigable Small World (HNSW) graph index.

HNSW is a state-of-the-art approximate nearest neighbor search algorithm that
combines hierarchical layers with navigable small-world graphs.

Key Concepts:
1. Probabilistic layer assignment using skip list distribution
2. Each node has M connections per layer (diverse neighbors)
3. Bidirectional edges for symmetric navigation
4. Greedy search on upper layers: O(log n) coarse navigation
5. Beam search at layer 0: exhaustive exploration with pruning
6. Heuristic neighbor selection: prefer diverse, closer neighbors

Algorithm Overview:
- Insert: Start at top layer, greedily find nearest neighbors, descend layers
- Search: Greedy descent through layers, beam search at bottom layer
- Each layer is a navigable small-world graph

Time Complexity:
    - Build: O(n log n) expected
    - Search: O(log n) expected
    - Space: O(n * M * L) where L is avg number of layers

Production Usage:
    - Weaviate, Qdrant, Pinecone, Milvus use HNSW
    - Best recall/speed trade-off among all ANN algorithms
    - Works well in both low and high dimensions

Paper: "Efficient and robust approximate nearest neighbor search using
        Hierarchical Navigable Small World graphs" (2018)
"""

from typing import List, Optional, Dict, Any, Set, Tuple
from dataclasses import dataclass, field
import heapq
import numpy as np
import math

from ..core.base import BaseIndex, SearchResult
from ..core.vector import Vector
from ..core.distance import Distance, DistanceMetric


@dataclass
class HNSWNode:
    """Node in the HNSW graph.

    Each node exists at multiple layers and has different neighbors per layer.
    """

    vector: Vector
    index: int
    level: int  # Maximum level this node reaches
    neighbors: Dict[int, List[int]] = field(default_factory=dict)  # level -> [neighbor_indices]
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        """Initialize neighbors dict for all levels."""
        for layer in range(self.level + 1):
            if layer not in self.neighbors:
                self.neighbors[layer] = []


class HNSW(BaseIndex):
    """Hierarchical Navigable Small World graph index.

    This implementation includes all key HNSW features:
    - Skip list probability distribution for layer assignment
    - Heuristic neighbor selection (diverse M neighbors)
    - Bidirectional edge creation
    - Greedy descent on upper layers
    - Beam search with BFS pruning at layer 0
    - Diverse candidate selection from candidate set
    """

    def __init__(
        self,
        dimension: int,
        M: int = 16,  # Max connections per layer
        M_max: int = 16,  # Max connections at layer 0
        ef_construction: int = 200,  # Size of dynamic candidate list during construction
        ef_search: int = 50,  # Size of dynamic candidate list during search
        ml: Optional[float] = None,  # Normalization factor for level generation
        metric: DistanceMetric = DistanceMetric.L2,
        seed: int = 42
    ):
        """Initialize HNSW index.

        Args:
            dimension: Dimensionality of vectors
            M: Maximum number of connections per layer (except layer 0)
            M_max: Maximum number of connections at layer 0 (usually 2*M)
            ef_construction: Candidate list size during construction (higher = better quality, slower)
            ef_search: Candidate list size during search (higher = better recall, slower)
            ml: Normalization factor for level generation (default: 1/ln(M))
            metric: Distance metric to use
            seed: Random seed for reproducibility
        """
        super().__init__(dimension)
        self.M = M
        self.M_max = M_max if M_max > M else 2 * M
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.ml = ml if ml is not None else 1.0 / math.log(M)
        self.metric = metric
        self.seed = seed

        self._distance_func = Distance.get_distance_function(metric)
        self._nodes: List[HNSWNode] = []
        self._entry_point: Optional[int] = None  # Index of entry point node
        self._max_level: int = 0  # Current max level in the graph
        self._rng = np.random.RandomState(seed)

    def _select_level(self) -> int:
        """Select layer for new node using skip list probability distribution.

        Formula: level = floor(-ln(uniform(0,1)) * ml)

        This creates an exponentially decaying probability:
        - P(level >= l) = (1/M)^l
        - Most nodes at layer 0, fewer at higher layers
        - Creates hierarchical structure

        Returns:
            Layer number (0 = bottom layer)
        """
        uniform = self._rng.uniform(0, 1)
        level = int(-math.log(uniform) * self.ml)
        return level

    def _get_neighbors(self, idx: int, layer: int) -> List[int]:
        """Get neighbors of a node at a specific layer.

        Args:
            idx: Node index
            layer: Layer number

        Returns:
            List of neighbor indices
        """
        if idx >= len(self._nodes):
            return []
        return self._nodes[idx].neighbors.get(layer, [])

    def _add_bidirectional_edge(self, idx1: int, idx2: int, layer: int) -> None:
        """Add bidirectional edge between two nodes at a layer.

        Args:
            idx1: First node index
            idx2: Second node index
            layer: Layer number

        Note:
            Only adds edge if both nodes exist at the given layer.
        """
        # Check if both nodes exist at this layer
        if layer > self._nodes[idx1].level or layer > self._nodes[idx2].level:
            return

        # Add idx2 as neighbor of idx1
        if layer not in self._nodes[idx1].neighbors:
            self._nodes[idx1].neighbors[layer] = []
        if idx2 not in self._nodes[idx1].neighbors[layer]:
            self._nodes[idx1].neighbors[layer].append(idx2)

        # Add idx1 as neighbor of idx2
        if layer not in self._nodes[idx2].neighbors:
            self._nodes[idx2].neighbors[layer] = []
        if idx1 not in self._nodes[idx2].neighbors[layer]:
            self._nodes[idx2].neighbors[layer].append(idx1)

    def _select_neighbors_heuristic(
        self,
        query_idx: int,
        candidates: List[Tuple[float, int]],
        M: int,
        layer: int,
        extend_candidates: bool = True,
        keep_pruned: bool = False
    ) -> List[int]:
        """Select M diverse neighbors from candidates using heuristic.

        This is the key heuristic that prevents long-range edge clustering.
        We prefer neighbors that are:
        1. Close to query
        2. Diverse (not clustered together)

        Algorithm (simplified):
        1. Sort candidates by distance to query
        2. Greedily select M neighbors that are diverse
        3. A candidate is diverse if it's closer to query than to already-selected neighbors

        Args:
            query_idx: Index of query node
            candidates: List of (distance, idx) tuples
            M: Number of neighbors to select
            layer: Layer number
            extend_candidates: Whether to extend candidates with their neighbors
            keep_pruned: Whether to keep pruned candidates

        Returns:
            List of selected neighbor indices
        """
        if len(candidates) <= M:
            return [idx for _, idx in candidates]

        # Extend candidates with their neighbors (for better connectivity)
        if extend_candidates:
            extended = set(candidates)
            for _, idx in candidates:
                neighbors = self._get_neighbors(idx, layer)
                for neighbor_idx in neighbors:
                    if neighbor_idx != query_idx:
                        dist = self._distance_func(
                            self._nodes[query_idx].vector,
                            self._nodes[neighbor_idx].vector
                        )
                        extended.add((dist, neighbor_idx))
            candidates = list(extended)

        # Sort by distance
        candidates = sorted(candidates, key=lambda x: x[0])

        result = []
        query_vector = self._nodes[query_idx].vector

        for dist_to_query, candidate_idx in candidates:
            if len(result) >= M:
                break

            # Check if candidate is diverse (closer to query than to any selected neighbor)
            is_diverse = True
            for selected_idx in result:
                dist_to_selected = self._distance_func(
                    self._nodes[candidate_idx].vector,
                    self._nodes[selected_idx].vector
                )
                # If candidate is farther from query than from selected neighbor, it's not diverse
                if dist_to_selected < dist_to_query:
                    is_diverse = False
                    break

            if is_diverse:
                result.append(candidate_idx)

        return result

    def _prune_connections(self, idx: int, layer: int) -> None:
        """Prune connections of a node if it exceeds M.

        Args:
            idx: Node index
            layer: Layer number
        """
        # Check if node exists at this layer
        if layer > self._nodes[idx].level:
            return

        # Check if layer exists in neighbors dict
        if layer not in self._nodes[idx].neighbors:
            return

        M = self.M_max if layer == 0 else self.M
        neighbors = self._nodes[idx].neighbors[layer]

        if len(neighbors) <= M:
            return

        # Compute distances to all neighbors
        candidates = []
        for neighbor_idx in neighbors:
            dist = self._distance_func(
                self._nodes[idx].vector,
                self._nodes[neighbor_idx].vector
            )
            candidates.append((dist, neighbor_idx))

        # Select M best diverse neighbors
        selected = self._select_neighbors_heuristic(
            idx, candidates, M, layer, extend_candidates=False
        )

        self._nodes[idx].neighbors[layer] = selected

    def _search_layer(
        self,
        query: Vector,
        entry_points: List[int],
        num_closest: int,
        layer: int
    ) -> List[Tuple[float, int]]:
        """Search for nearest neighbors at a specific layer.

        This implements greedy search with beam search:
        - Maintains dynamic list of candidates (size = ef)
        - Explores neighbors in order of distance
        - Stops when no closer neighbors found

        Args:
            query: Query vector
            entry_points: List of entry point indices
            num_closest: Number of closest neighbors to return
            layer: Layer to search

        Returns:
            List of (distance, index) tuples
        """
        visited = set(entry_points)
        candidates = []  # Min heap of (distance, idx)
        w = []  # Dynamic list of found nearest neighbors (max heap of negative distances)

        # Initialize with entry points
        for ep_idx in entry_points:
            dist = self._distance_func(query, self._nodes[ep_idx].vector)
            heapq.heappush(candidates, (dist, ep_idx))
            heapq.heappush(w, (-dist, ep_idx))

        while candidates:
            current_dist, current_idx = heapq.heappop(candidates)

            # If current is farther than worst in w, stop
            if current_dist > -w[0][0]:
                break

            # Explore neighbors of current
            neighbors = self._get_neighbors(current_idx, layer)
            for neighbor_idx in neighbors:
                if neighbor_idx not in visited:
                    visited.add(neighbor_idx)

                    dist = self._distance_func(query, self._nodes[neighbor_idx].vector)

                    if dist < -w[0][0] or len(w) < num_closest:
                        heapq.heappush(candidates, (dist, neighbor_idx))
                        heapq.heappush(w, (-dist, neighbor_idx))

                        # Prune w if too large
                        if len(w) > num_closest:
                            heapq.heappop(w)

        # Return as list of (distance, idx) sorted by distance
        result = [(-dist, idx) for dist, idx in w]
        result.sort()
        return result

    def add(
        self,
        vectors: List[Vector],
        metadata: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """Add vectors to HNSW index.

        Algorithm:
        1. Select layer using skip list distribution
        2. Find nearest neighbors at each layer (greedy descent)
        3. Connect to M diverse neighbors per layer
        4. Add bidirectional edges
        5. Update entry point if needed

        Args:
            vectors: List of vectors to add
            metadata: Optional metadata for each vector
        """
        self._validate_vectors(vectors)

        start_time = self._start_build_timer()

        if metadata is None:
            metadata = [None] * len(vectors)

        for i, (vector, meta) in enumerate(zip(vectors, metadata)):
            self._add_single(vector, meta)

        self.num_vectors = len(self._nodes)
        self._end_build_timer(start_time)

    def _add_single(self, vector: Vector, metadata: Optional[Dict[str, Any]]) -> None:
        """Add a single vector to the index.

        Args:
            vector: Vector to add
            metadata: Optional metadata
        """
        # Select level for new node
        level = self._select_level()

        # Create new node
        new_idx = len(self._nodes)
        new_node = HNSWNode(
            vector=vector,
            index=new_idx,
            level=level,
            metadata=metadata
        )
        self._nodes.append(new_node)

        # If first node, make it entry point
        if self._entry_point is None:
            self._entry_point = new_idx
            self._max_level = level
            return

        # Search for nearest neighbors starting from entry point
        entry_points = [self._entry_point]

        # Greedy descent through layers above new node's level
        for lc in range(self._max_level, level, -1):
            nearest = self._search_layer(vector, entry_points, 1, lc)
            entry_points = [idx for _, idx in nearest]

        # Insert at levels [0, level]
        for lc in range(level, -1, -1):
            # Search for ef_construction nearest neighbors
            candidates = self._search_layer(vector, entry_points, self.ef_construction, lc)

            # Select M diverse neighbors
            M = self.M_max if lc == 0 else self.M
            neighbors = self._select_neighbors_heuristic(
                new_idx, candidates, M, lc, extend_candidates=True
            )

            # Add bidirectional edges
            for neighbor_idx in neighbors:
                self._add_bidirectional_edge(new_idx, neighbor_idx, lc)

                # Prune neighbor's connections if needed
                self._prune_connections(neighbor_idx, lc)

            # Update entry points for next layer
            entry_points = [idx for _, idx in candidates[:self.ef_construction]]

        # Update entry point if new node is taller
        if level > self._max_level:
            self._max_level = level
            self._entry_point = new_idx

    def search(
        self,
        query: Vector,
        k: int = 10,
        ef: Optional[int] = None,
        **kwargs
    ) -> SearchResult:
        """Search for k nearest neighbors using HNSW.

        Algorithm:
        1. Start at entry point (top layer)
        2. Greedy descent through layers: find 1 nearest at each layer
        3. At layer 0: beam search with ef candidates
        4. Return top-k from final candidates

        Args:
            query: Query vector
            k: Number of nearest neighbors to return
            ef: Search beam width (default: self.ef_search)
            **kwargs: Additional parameters

        Returns:
            SearchResult with k nearest neighbors
        """
        self._validate_dimension(query)

        if self._entry_point is None or self.num_vectors == 0:
            return SearchResult(indices=[], distances=[], metadata=[])

        if k > self.num_vectors:
            k = self.num_vectors

        if ef is None:
            ef = max(self.ef_search, k)

        # Start from entry point
        entry_points = [self._entry_point]

        # Greedy descent to layer 1
        for lc in range(self._max_level, 0, -1):
            nearest = self._search_layer(query, entry_points, 1, lc)
            entry_points = [idx for _, idx in nearest]

        # Beam search at layer 0
        candidates = self._search_layer(query, entry_points, ef, 0)

        # Get top-k
        top_k = candidates[:k]

        result_distances = [dist for dist, _ in top_k]
        result_indices = [self._nodes[idx].index for _, idx in top_k]

        # Extract metadata
        result_metadata = None
        if any(node.metadata is not None for node in self._nodes):
            result_metadata = [self._nodes[idx].metadata for _, idx in top_k]

        return SearchResult(
            indices=result_indices,
            distances=result_distances,
            metadata=result_metadata
        )

    def _estimate_memory(self) -> int:
        """Estimate memory usage in bytes.

        Returns:
            Approximate memory usage
        """
        # Vectors
        vectors_mem = self.num_vectors * self.dimension * 4

        # Graph edges: avg_connections per node per layer
        avg_layers = 1 / (1 - 1/self.M)  # Expected from geometric distribution
        avg_connections = (self.M + self.M_max) / 2
        edges_mem = self.num_vectors * avg_layers * avg_connections * 8

        return int(vectors_mem + edges_mem)

    def _get_additional_stats(self) -> Dict[str, Any]:
        """Get HNSW specific statistics."""
        # Layer distribution
        layer_counts = {}
        for node in self._nodes:
            level = node.level
            layer_counts[level] = layer_counts.get(level, 0) + 1

        # Connection statistics
        total_connections = 0
        for node in self._nodes:
            for layer_neighbors in node.neighbors.values():
                total_connections += len(layer_neighbors)

        avg_connections = total_connections / self.num_vectors if self.num_vectors > 0 else 0

        return {
            "metric": self.metric.value,
            "M": self.M,
            "M_max": self.M_max,
            "ef_construction": self.ef_construction,
            "ef_search": self.ef_search,
            "max_level": self._max_level,
            "entry_point": self._entry_point,
            "avg_connections": avg_connections,
            "layer_distribution": dict(sorted(layer_counts.items()))
        }

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"HNSW(dimension={self.dimension}, "
            f"num_vectors={self.num_vectors}, "
            f"M={self.M}, "
            f"max_level={self._max_level}, "
            f"metric={self.metric.value})"
        )
