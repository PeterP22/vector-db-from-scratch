# Vector Database from Scratch

A complete implementation of a vector database built from first principles, featuring multiple indexing algorithms, compression techniques, and optimization strategies.

## Features

### Search Algorithms
- **Linear Scan**: Brute force baseline with 100% recall
- **KD-Tree**: Binary space partitioning with dimension cycling and branch pruning
- **LSH (Locality Sensitive Hashing)**: Random projection with multi-table strategy
- **HNSW (Hierarchical Navigable Small World)**: State-of-the-art graph-based search
- **IVF (Inverted File Index)**: K-means clustering with multi-probe search

### Compression & Optimization
- **Scalar Quantization**: 4x compression (float32 → uint8)
- **Product Quantization**: 16-64x compression with subspace clustering
- **SIMD Optimizations**: Vectorized distance computations (4-37x speedup)
- **Asymmetric Distance Computation (ADC)**: Efficient PQ search

### Core Features
- Distance metrics: L2/Euclidean, Cosine Similarity, Dot Product/MIPS
- Metadata support for all indexes
- Comprehensive benchmarking and statistics
- Production-ready architecture with base classes

## Quick Start

### 1. Run the Document Demo

```bash
python examples/document_demo.py
```

This demo shows:
- Document ingestion with embedding generation
- Building indexes with all 5 algorithms
- Natural language query search
- Performance comparison across algorithms

### 2. Use Individual Indexes

```python
from src.indexes.hnsw import HNSW
import numpy as np

# Create index
index = HNSW(dimension=128, M=16, ef_construction=200)

# Add vectors with metadata
vectors = [np.random.randn(128).astype(np.float32) for _ in range(1000)]
metadata = [{"id": i, "text": f"Document {i}"} for i in range(1000)]
index.add(vectors, metadata)

# Search
query = np.random.randn(128).astype(np.float32)
result = index.search(query, k=10)

# Access results
for idx, dist, meta in zip(result.indices, result.distances, result.metadata):
    print(f"Document {meta['id']}: distance={dist:.4f}")
```

### 3. Use Quantization

```python
from src.quantization.scalar import ScalarQuantizer
from src.quantization.product import ProductQuantizer

# Scalar Quantization (4x compression)
quantizer = ScalarQuantizer(dimension=128)
quantizer.train(vectors)
quantized = quantizer.encode_batch(vectors)  # uint8 vectors

# Product Quantization (64x compression)
pq = ProductQuantizer(dimension=128, num_subspaces=8, num_clusters=256)
pq.train(vectors)
codes = pq.encode_batch(vectors)  # 8 bytes per vector

# Fast search with ADC
query = vectors[0]
distance_table = pq.compute_distance_table(query)
distances = pq.adc_distance_batch(codes, distance_table)
```

## Project Structure

```
vector-db-from-scratch/
├── src/
│   ├── core/               # Core utilities
│   │   ├── base.py         # Base index interface
│   │   ├── distance.py     # Distance metrics
│   │   └── vector.py       # Vector operations
│   ├── indexes/            # Search algorithms
│   │   ├── linear_scan.py  # Linear scan
│   │   ├── kdtree.py       # KD-Tree
│   │   ├── lsh.py          # LSH
│   │   ├── hnsw.py         # HNSW
│   │   └── ivf.py          # IVF
│   ├── quantization/       # Compression
│   │   ├── scalar.py       # Scalar quantization
│   │   ├── product.py      # Product quantization
│   │   └── simd_ops.py     # SIMD operations
│   └── utils/              # Utilities
│       └── data_generator.py
├── examples/
│   └── document_demo.py    # End-to-end demo
├── PLAN.md                 # Detailed implementation plan
└── README.md              # This file
```

## Algorithm Comparison

### Performance Characteristics

| Algorithm | Build Time | Query Time | Recall | Memory | Best For |
|-----------|-----------|-----------|--------|--------|----------|
| Linear Scan | O(1) | O(n) | 100% | 1x | Small datasets (<10K) |
| KD-Tree | O(n log n) | O(log n) - O(n) | 100% | 1x | Low dimensions (<8D) |
| LSH | O(n) | O(n/L) | 70-90% | 2-5x | Very large datasets |
| HNSW | O(n log n) | O(log n) | 95-99% | 2-3x | **Production (best overall)** |
| IVF | O(n×k) | O(n/k) | 85-95% | 1.1x | Billion-scale with PQ |

### Use Case Recommendations

**Small Datasets (<100K vectors)**
- Use: **HNSW**
- Why: Excellent recall, fast queries, easy to use

**Large Datasets (1M-100M vectors)**
- Use: **IVF + Product Quantization**
- Why: Sublinear query time, extreme compression

**Extremely Large (>100M vectors)**
- Use: **HNSW + Scalar Quantization** or **IVF + PQ**
- Why: Balance of speed, memory, and accuracy

**Real-time Updates**
- Use: **HNSW**
- Why: Supports efficient incremental inserts

**Batch Updates**
- Use: **IVF**
- Why: Can rebuild clusters periodically

## Key Concepts Demonstrated

### 1. Curse of Dimensionality
KD-Tree works well in low dimensions but degrades to linear scan in high dimensions:
- 4D: ~85% pruning efficiency
- 8D: ~40% pruning efficiency
- 16D+: ~0% pruning efficiency (no better than linear scan)

### 2. Approximate vs Exact Search
- **Exact**: Linear scan, KD-Tree (low-D) - 100% recall but slow
- **Approximate**: LSH, HNSW, IVF - 90-99% recall but much faster

### 3. Space Partitioning Strategies
- **Tree-based**: KD-Tree (binary space partitioning)
- **Hash-based**: LSH (locality-sensitive hashing)
- **Graph-based**: HNSW (navigable small world)
- **Cluster-based**: IVF (Voronoi cells)

### 4. Compression Trade-offs
- **Scalar Quantization**: 4x compression, ~1% error, very fast
- **Product Quantization**: 16-64x compression, ~5-10% error, ADC is fast
- **Binary Quantization**: 32x compression, ~15% error, extreme speed

## Benchmarks

From the document demo with 15 documents (128D vectors):

```
Index           Build(s)     Query(ms)    QPS
---------------------------------------------------
Linear Scan     0.0000       0.02         43,564
KD-Tree         0.0001       0.05         21,456
LSH             0.0018       0.07         13,854
HNSW            0.0005       0.03         28,693
IVF             0.0005       0.02         46,152
```

Note: On larger datasets (>100K vectors), HNSW and IVF significantly outperform linear scan.

## Production Considerations

### What's Implemented
- All core algorithms from scratch
- Multiple distance metrics
- Metadata support
- Quantization techniques
- SIMD optimizations

### What's Missing (for production)
- Persistence (save/load indexes)
- Metadata filtering
- Deletion and updates
- Multi-threading
- GPU acceleration
- Distributed search
- API layer (REST/gRPC)

### Recommended Production Systems
- **Weaviate**: HNSW-based, full-featured
- **Qdrant**: Rust-based, fast filtering
- **Pinecone**: Managed service
- **Milvus**: Supports multiple indexes
- **pgvector**: PostgreSQL extension
- **FAISS**: C++ library (Meta AI)

## Implementation Highlights

### HNSW Key Features (src/indexes/hnsw.py:64)
- Skip list probability distribution for layer assignment
- Diverse neighbor selection (M neighbors)
- Bidirectional edges
- Greedy descent on upper layers
- Beam search at layer 0

### Product Quantization (src/quantization/product.py:63)
- Subspace k-means clustering
- Asymmetric Distance Computation (ADC)
- 64x compression with codebooks
- Precomputed distance tables for fast search

### IVF Implementation (src/indexes/ivf.py:230)
- K-means++ initialization
- Voronoi cell assignment
- Multi-probe search
- Inverted lists structure

## Testing

Each implementation includes comprehensive tests:

```bash
# Test scalar quantization
python src/quantization/scalar.py

# Test product quantization
python src/quantization/product.py

# Test SIMD speedups
python src/quantization/simd_ops.py

# Test document demo
python examples/document_demo.py
```

## References

This implementation is based on research and production systems:

- **HNSW**: Malkov & Yashunin (2018) - "Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs"
- **IVF**: Jégou et al. (2011) - "Product quantization for nearest neighbor search"
- **LSH**: Charikar (2002) - "Similarity estimation techniques from rounding algorithms"
- **Production Systems**: FAISS, Weaviate, Qdrant, Milvus, pgvector

## Next Steps

1. **Use Real Embeddings**: Replace `SimpleEmbedder` with sentence-transformers or OpenAI
2. **Scale Up**: Test with larger datasets (100K-1M vectors)
3. **Add Persistence**: Implement save/load for indexes
4. **Metadata Filtering**: Add pre/post filtering support
5. **API Layer**: Wrap in FastAPI for REST interface
6. **Benchmarking**: Compare against FAISS or Annoy

## License

MIT - Built for educational purposes to understand vector database internals from first principles.
