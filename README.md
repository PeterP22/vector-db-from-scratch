# Vector Database from Scratch

A complete production-grade vector database implementation from first principles, featuring 5 search algorithms, state-of-the-art embeddings, and LLM integration for RAG applications.

## 🎯 What Makes This Special

- **Built from Scratch**: Every algorithm implemented from first principles for deep understanding
- **Production Embeddings**: Nvidia Nemotron models for high-quality vector representations
- **LLM Integration**: Kimi K2 (Chinese LLM) for intelligent answer synthesis
- **GPU Accelerated**: Mac Metal (MPS) and CUDA support for fast inference
- **Complete RAG System**: Novel Q&A with streaming responses and smart caching
- **Educational**: Extensive documentation showing how everything works under the hood

## 🚀 Features

### Search Algorithms (5 Implementations)
- **Linear Scan**: Brute force baseline with 100% recall
- **KD-Tree**: Binary space partitioning with dimension cycling and branch pruning
- **LSH (Locality Sensitive Hashing)**: Random projection with multi-table strategy
- **HNSW (Hierarchical Navigable Small World)**: State-of-the-art graph-based search
- **IVF (Inverted File Index)**: K-means clustering with multi-probe search

### Advanced Embeddings & Re-ranking
- **Nvidia Nemotron Integration**:
  - `llama-3.2-nv-embedqa-1b-v2` (1B params, fast embeddings)
  - `llama-embed-nemotron-8b` (8B params, best quality embeddings)
  - `llama-3.2-nv-rerankqa-1b-v2` (1B params, re-ranking)
  - 2048-4096 dimensional embeddings
  - GPU acceleration (4x faster than CPU)
  - FP16 inference (2x faster, 50% less memory)
  - **Safetensors optimization** (20x faster model loading: 3s vs 60s)
  - Two-stage retrieval: Vector search → Re-ranking

### LLM Integration
- **Kimi K2 Models**:
  - `kimi-k2-turbo-preview` (60-100 tokens/s, high-speed)
  - `kimi-k2-0905-preview` (256K context window)
  - Streaming responses for real-time interaction
  - OpenAI-compatible API
  - ⚠️ Web search (experimental - requires multi-turn tool call handling)

### Compression & Optimization
- **Scalar Quantization**: 4x compression (float32 → uint8)
- **Product Quantization**: 16-64x compression with subspace clustering
- **SIMD Optimizations**: Vectorized distance computations (4-37x speedup)
- **Asymmetric Distance Computation (ADC)**: Efficient PQ search
- **Smart Caching**: Embeddings cached to disk for instant reload
- **FP16 Inference**: Half-precision on GPU for 2x faster processing
- **Safetensors Loading**: Memory-mapped loading for 20x faster model startup

### Production Features
- **GPU Acceleration**: Mac Metal (MPS), NVIDIA CUDA support
- **Interactive RAG**: Novel Q&A with streaming LLM responses
- **Metadata Support**: Rich metadata filtering and search
- **Comprehensive Benchmarking**: Performance metrics for all algorithms
- **Production Architecture**: Clean base classes and modular design

## Quick Start

### 1. Novel RAG with Kimi LLM (Recommended) ⭐

The complete RAG experience with GPU acceleration, streaming responses, and intelligent answers:

```bash
# Install dependencies
pip install torch transformers pypdf openai python-dotenv

# Set up your API keys in .env
echo "KIMI_API_KEY=your_api_key_here" >> .env

# Run interactive RAG
python examples/4_complete_rag_system.py
```

**Features:**
- ✅ Nvidia embeddings with Mac GPU acceleration + FP16
- ✅ Two-stage retrieval: Vector search (15 candidates) → Re-ranking (top 5)
- ✅ Safetensors optimization (3-second model loading)
- ✅ Kimi K2 LLM for synthesized answers
- ✅ Streaming responses (see answers appear word-by-word)
- ✅ Smart caching (instant reload after first run)
- ✅ Interactive Q&A mode with commands ('raw', 'quit')
- ⚠️ Web search (experimental - requires multi-turn tool handling)

**Example:**
```
📖 Question: what is the main theme of the book?

🤖 Kimi's Answer (streaming...)

Based on the passages from your novel, the main theme revolves around...
[answer streams in real-time, citing specific pages]

⏱️  Search: 155ms | Re-rank: 2.4s | LLM: 5.2s | Total: 7.8s
```

#### Nvidia Re-ranker Requirements
- **Trust remote code**: the Nemotron reranker ships a custom `llama_bidirectional_model` implementation. `examples/4_complete_rag_system.py` automatically enables `trust_remote_code=True`, but Hugging Face will prompt the first time—answer “yes” to allow the custom module to run.
- **Allow large downloads**: the reranker checkpoint is ~2.5 GB. Make sure you have disk space in `~/.cache/huggingface`.
- **GPU strongly recommended**: works on Apple MPS and CUDA. You can set `use_gpu=False` in `NvidiaReranker` if you must fall back to CPU, but latency will be much higher.
- **Optional Hugging Face login**: if the NVIDIA repo is gated for your account, run `huggingface-cli login` once before launching the example.

### 2. Learn the Basics - Algorithm Comparison

```bash
python examples/1_basic_index_comparison.py
```

This demo shows:
- Document ingestion with simple embedding generation
- Building indexes with all 5 algorithms (Linear, KD-Tree, LSH, HNSW, IVF)
- Natural language query search
- Performance comparison and benchmarks

### 3. Use Individual Indexes

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

### 4. Use Quantization

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

## 📁 Project Structure

```
vector-db-from-scratch/
├── examples/                      # Learning path: 1 → 2 → 3 → 4
│   ├── 1_basic_index_comparison.py    # All 5 algorithms comparison
│   ├── 2_nvidia_embeddings_demo.py    # Production embeddings
│   ├── 3_rag_retrieval_only.py        # RAG without LLM synthesis
│   ├── 4_complete_rag_system.py       # Full RAG with streaming LLM ⭐
│   └── NVIDIA_SETUP.md                # Setup guide for Nvidia models
├── prompts/
│   ├── system_prompt.txt         # LLM system prompt (easily customizable)
│   ├── user_prompt_template.txt  # User query template
│   └── README.md                 # Prompt engineering guide
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
- GPU acceleration (Mac MPS + NVIDIA CUDA)
- Nvidia Nemotron embeddings
- Nvidia re-ranking model (two-stage retrieval)
- Kimi K2 LLM integration
- Streaming responses
- Smart caching system
- Complete RAG pipeline with re-ranking

### What's Missing (for production)
- Persistence (save/load indexes)
- Metadata filtering
- Deletion and updates
- Multi-threading
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

## Learning Path

The examples are numbered to guide you from basics to production:

**1️⃣ Start Here** - Compare all 5 search algorithms:
```bash
python examples/1_basic_index_comparison.py
```

**2️⃣ Production Embeddings** - Learn Nvidia Nemotron integration:
```bash
python examples/2_nvidia_embeddings_demo.py
```

**3️⃣ RAG Retrieval** - Build retrieval without LLM synthesis:
```bash
python examples/3_rag_retrieval_only.py
```

**4️⃣ Complete RAG** - Full system with streaming LLM:
```bash
python examples/4_complete_rag_system.py
```

## Testing

Individual components include test functions:

```bash
# Test scalar quantization
python src/quantization/scalar.py

# Test product quantization
python src/quantization/product.py

# Test SIMD speedups
python src/quantization/simd_ops.py
```

## References

This implementation is based on research and production systems:

- **HNSW**: Malkov & Yashunin (2018) - "Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs"
- **IVF**: Jégou et al. (2011) - "Product quantization for nearest neighbor search"
- **LSH**: Charikar (2002) - "Similarity estimation techniques from rounding algorithms"
- **Nvidia Nemotron**: [NVIDIA RAG Collection](https://huggingface.co/collections/nvidia/nemotron-rag)
- **Kimi K2**: [Moonshot AI Platform](https://platform.moonshot.ai/docs/guide/kimi-k2-quickstart)
- **Production Systems**: FAISS, Weaviate, Qdrant, Milvus, pgvector

## Next Steps

1. ~~**Use Real Embeddings**~~: ✅ Implemented with Nvidia Nemotron models
2. ~~**LLM Integration**~~: ✅ Implemented with Kimi K2 streaming responses
3. **Scale Up**: Test with larger datasets (100K-1M vectors)
4. **Add Persistence**: Implement save/load for indexes
5. **Metadata Filtering**: Add pre/post filtering support
6. **API Layer**: Wrap in FastAPI for REST interface
7. **Benchmarking**: Compare against FAISS or Annoy

## License

MIT - Built for educational purposes to understand vector database internals from first principles.
