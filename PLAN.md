# Vector Database From Scratch - Learning Plan

**Goal:** Build a production-grade vector database from first principles to understand all core techniques used in modern vector search systems (Pinecone, Weaviate, Qdrant, Milvus).

## Project Overview

This project implements multiple indexing strategies and optimizations:
- **Basic:** Linear scan baseline, distance metrics
- **Tree-based:** KD-Tree with optimized pruning
- **Hash-based:** LSH with random projections
- **Graph-based:** HNSW with hierarchical navigation
- **Cluster-based:** IVF with k-means quantization
- **Compression:** Scalar, product, and binary quantization
- **Production:** Persistence, filtering, deletions, concurrency, SIMD

---

## Phase 1: Foundation (Tasks 1-3)

### Task 1: Set up project structure and core utilities
- Vector operations (add, subtract, scale)
- Test data generation (random, clustered, high-dimensional)
- Base interfaces for all index types

### Task 2: Implement distance metrics
- **L2/Euclidean distance:** `sqrt(sum((a[i] - b[i])^2))`
- **Cosine similarity:** `dot(a,b) / (||a|| * ||b||)`
- **Dot product/MIPS:** Maximum Inner Product Search
- Understand when to use each metric

### Task 3: Linear scan baseline
- Brute force O(n) search through all vectors
- Benchmark baseline for recall (100%) and latency
- Critical for comparing all approximate methods

**Key Learning:** Distance metrics affect everything downstream. Cosine similarity needs normalization, dot product is asymmetric.

---

## Phase 2: KD-Tree (Tasks 4-7)

### Task 4: Recursive BST insertion with d-cycling
- Binary space partitioning in d dimensions
- Split dimension cycles: depth % d
- Recursive tree construction

### Task 5: Hyperplane perpendicular splitting
- Split perpendicular to axis at depth % d
- Median-based partitioning for balanced tree
- Understand curse of dimensionality (KD-trees fail at d>20)

### Task 6: Bound and branch pruning with greedy DFS
- Bounding box intersection tests
- Greedy depth-first search prioritizing closer branches
- Track best-k candidates and prune impossible branches

### Task 7: Optimize pruning by ~50%
- Early termination strategies
- Better bounding calculations
- Measure node visits before/after optimization

**Key Learning:** Tree-based methods work well in low dimensions but degrade exponentially with dimensionality. Pruning is critical for performance.

---

## Phase 3: LSH - Locality Sensitive Hashing (Tasks 8-9)

### Task 8: LSH with random projection (SimHash)
- Generate random hyperplanes (unit vectors)
- Hash each vector: `h[i] = 1 if dot(v, r[i]) > 0 else 0`
- Similar vectors → same hash bucket (high probability)
- **Paradigm shift:** Hash collisions are DESIRED

### Task 9: Multi-table strategy and bucket search
- Use L tables with different random projections
- Query searches all L buckets and merges results
- Trade-off: more tables = higher recall, more memory

**Key Learning:** LSH is probabilistic and uses completely different principles than tree/graph methods. Great for very high dimensions where trees fail.

---

## Phase 4: HNSW - Hierarchical Navigable Small World (Tasks 10-15)

### Task 10: Probabilistic layer assignment (skip list distribution)
- Assign each node to layers using: `layer = floor(-ln(uniform(0,1)) * ml)`
- `ml = 1/ln(M)` where M is max connections
- Creates hierarchical structure (fewer nodes at higher layers)

### Task 11: Neighbor selection heuristic (M diverse neighbors)
- Select M neighbors that are diverse (not clustered)
- Prevents long-range edge clustering
- Heuristic considers both distance and angular diversity

### Task 12: Bidirectional edge creation
- When inserting node, add edges FROM new node
- Also add edges TO new node from neighbors
- Maintains graph connectivity

### Task 13: Greedy descent on upper layers
- Start at highest layer, navigate to nearest neighbor
- Drop to next layer, repeat
- O(log n) coarse search to get close to target

### Task 14: Exhaustive beam search at layer 0
- Maintain candidate set (size = ef)
- BFS exploration with pruning
- Examines all promising neighbors at base layer

### Task 15: Heuristic for diverse M candidates
- From candidate set, pick M diverse neighbors
- Considers distance + diversity to prevent clustering
- Critical for maintaining graph quality

**Key Learning:** HNSW combines skip list structure with small-world graphs. The hierarchical approach gives O(log n) complexity while maintaining high recall.

---

## Phase 5: IVF - Inverted File Index (Tasks 16-17)

### Task 16: K-means clustering for coarse quantization
- Cluster vectors into k partitions (Voronoi cells)
- Each cluster has a centroid
- Coarse quantization: find nearest centroid(s)

### Task 17: Inverted lists and multi-probe search
- Each cluster has inverted list of vectors
- Multi-probe: search top-k nearest clusters
- Trade-off: more probes = higher recall, slower search

**Key Learning:** IVF is the coarse quantization stage in many production systems (FAISS). Reduces search space from n vectors to n/k vectors per cluster.

---

## Phase 6: Quantization Techniques (Tasks 18-22)

### Task 18: Scalar quantization (float32 → uint8)
- Map float32 range to 0-255 (uint8)
- 4x memory compression
- Enables SIMD optimizations

### Task 19: SIMD optimizations for distance calculations
- Use CPU SIMD instructions (AVX2, AVX-512)
- Parallel distance computations on quantized vectors
- 4x+ speedup in practice

### Task 20: Product quantization (PQ)
- Split d-dimensional vector into m subspaces
- K-means cluster each subspace independently
- Store cluster IDs instead of raw values
- Massive compression: d*4 bytes → m*1 byte

### Task 21: Asymmetric distance computation (ADC)
- Query vector stays full precision
- Database vectors are quantized
- Distance = sum of distances to subspace centroids
- Better accuracy than symmetric distance

### Task 22: Binary quantization (1-bit per dimension)
- Each dimension: 1 if positive, 0 if negative
- 32x compression (float32 → 1 bit)
- Hamming distance for ultra-fast comparison

**Key Learning:** Quantization is CRITICAL for production systems. You can't fit billions of float32 vectors in memory. PQ+SIMD is the standard approach.

---

## Phase 7: Production Features (Tasks 23-26)

### Task 23: Metadata filtering
- Attach metadata to each vector
- Filtered search: "find similar WHERE category='X'"
- Pre-filtering vs post-filtering trade-offs

### Task 24: Persistence layer
- Serialize/deserialize indexes to disk
- Memory-mapped files for large indexes
- Efficient loading without full memory copy

### Task 25: Deletion support
- Tombstoning: mark deleted, don't remove
- Lazy removal: rebuild index periodically
- Maintain graph connectivity after deletions

### Task 26: Update operations
- Update = delete + reinsert
- Optimization: reuse connections when possible
- Batch updates for efficiency

**Key Learning:** Production systems need more than just search. Deletions are surprisingly complex for graph structures.

---

## Phase 8: Hybrid Approaches (Tasks 27-28)

### Task 27: HNSW+PQ (compressed vectors in-graph)
- Store PQ-compressed vectors in HNSW graph
- Use ADC for distance calculations during search
- Weaviate-style implementation

### Task 28: IVF+PQ (FAISS standard combination)
- IVF for coarse search (find clusters)
- PQ for fine search within clusters
- The standard production approach in FAISS

**Key Learning:** Real systems combine techniques. IVF+PQ is the gold standard for billion-scale search.

---

## Phase 9: Concurrency (Tasks 29-30)

### Task 29: Multi-threading for index construction
- Parallel insertion where possible
- Lock-based or lock-free approaches
- Speedup index building by CPU core count

### Task 30: Concurrent query support
- Thread-safe read operations
- Read-write locks for updates during queries
- Benchmark QPS (queries per second)

**Key Learning:** Concurrency is essential for production. Modern CPUs have many cores - use them!

---

## Phase 10: Evaluation (Tasks 31-32)

### Task 31: Comprehensive benchmark suite
- **Recall@k:** % of true nearest neighbors found
- **QPS:** Queries per second (throughput)
- **Latency:** P50, P95, P99 query times
- **Memory:** Index size vs raw vectors
- **Build time:** Index construction time
- Compare all methods across dimensions, dataset sizes

### Task 32: Visualization tools
- Index structure visualization (graphs, trees)
- Search path visualization (how queries navigate)
- Performance dashboards (recall vs latency curves)
- Distance distribution plots

**Key Learning:** Proper evaluation is critical. All ANN algorithms trade recall for speed. Understand the trade-offs.

---

## Key Concepts Summary

### Algorithm Trade-offs
| Algorithm | Complexity | Best For | Weakness |
|-----------|-----------|----------|----------|
| Linear Scan | O(n) | Small datasets, 100% recall | Doesn't scale |
| KD-Tree | O(log n) ideal | Low dimensions (d<20) | Curse of dimensionality |
| LSH | O(1) hash lookup | Very high dimensions | Needs many tables for recall |
| HNSW | O(log n) | General purpose, high recall | Memory intensive |
| IVF | O(n/k) per cluster | With PQ for compression | Accuracy depends on clustering |

### Compression Comparison
| Method | Compression | Speed | Accuracy |
|--------|-------------|-------|----------|
| None | 1x | Fast | Perfect |
| Scalar | 4x | Very fast (SIMD) | ~95% |
| Product | 16-32x | Medium | ~90% |
| Binary | 32x | Ultra fast | ~85% |

### Production Considerations
1. **Quantization is mandatory** for billion-scale systems
2. **HNSW** is state-of-the-art for recall/speed trade-off
3. **IVF+PQ** is most memory-efficient for large scale
4. **SIMD** optimizations give real 4x+ speedups
5. **Hybrid approaches** (HNSW+PQ) combine benefits

---

## References & Resources

### Papers
- HNSW: "Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs" (2018)
- DiskANN: "Fast Accurate Billion-point Nearest Neighbor Search on a Single Node" (2019)
- Product Quantization: "Product quantization for nearest neighbor search" (2011)
- LSH: "Locality-Sensitive Hashing Scheme Based on p-Stable Distributions" (2004)

### Production Systems
- FAISS (Meta): IVF+PQ standard implementation
- Weaviate: HNSW+quantization
- Qdrant: HNSW with filtering
- Pinecone: Proprietary (likely HNSW-based)
- Milvus: Multiple index types

### What We're Building
A complete vector database that demonstrates every major technique used in production systems, from first principles. By the end, you'll understand why Pinecone makes different choices than FAISS, why HNSW dominates, and when to use each technique.

---

**Status:** Planning complete. Ready to start Phase 1.

**Last Updated:** 2025-11-03
