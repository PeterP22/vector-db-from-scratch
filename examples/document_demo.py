"""Document Ingestion and Query Demo.

This demo shows a complete end-to-end workflow:
1. Load documents from text files
2. Generate embeddings (using sentence-transformers or mock embeddings)
3. Build indexes using different algorithms (Linear, KD-Tree, LSH, HNSW, IVF)
4. Run natural language queries
5. Compare performance and recall across algorithms

Production Note:
    In production, you'd use:
    - sentence-transformers, OpenAI embeddings, or Cohere embeddings
    - Persistent storage (SQLite, PostgreSQL, Redis)
    - Metadata filtering and hybrid search
    - API layer (FastAPI, Flask)
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from typing import List, Dict, Tuple
import time
from dataclasses import dataclass

from src.core.vector import Vector
from src.indexes.linear_scan import LinearScan
from src.indexes.kdtree import KDTree
from src.indexes.lsh import LSH
from src.indexes.hnsw import HNSW
from src.indexes.ivf import IVF


@dataclass
class Document:
    """Document with text and metadata."""
    id: str
    text: str
    metadata: Dict


class SimpleEmbedder:
    """Simple embedding generator for demo purposes.

    In production, use:
    - sentence-transformers: all-MiniLM-L6-v2, all-mpnet-base-v2
    - OpenAI: text-embedding-3-small, text-embedding-ada-002
    - Cohere: embed-english-v3.0

    This creates deterministic embeddings based on word hashing for demo.
    """

    def __init__(self, dimension: int = 128):
        self.dimension = dimension
        self._word_cache = {}

    def _word_to_vector(self, word: str) -> Vector:
        """Convert word to vector using hash-based approach."""
        if word in self._word_cache:
            return self._word_cache[word]

        # Use word hash to seed random generator for reproducibility
        seed = hash(word) % (2**32)
        rng = np.random.RandomState(seed)
        vector = rng.randn(self.dimension).astype(np.float32)

        # Normalize
        vector = vector / np.linalg.norm(vector)

        self._word_cache[word] = vector
        return vector

    def embed(self, text: str) -> Vector:
        """Generate embedding for text by averaging word vectors."""
        # Simple tokenization (split on whitespace and lowercase)
        words = text.lower().split()

        if not words:
            return np.zeros(self.dimension, dtype=np.float32)

        # Average word vectors
        vectors = [self._word_to_vector(word) for word in words]
        embedding = np.mean(vectors, axis=0).astype(np.float32)

        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding

    def embed_batch(self, texts: List[str]) -> List[Vector]:
        """Generate embeddings for multiple texts."""
        return [self.embed(text) for text in texts]


def load_sample_documents() -> List[Document]:
    """Load sample documents about various topics."""
    documents = [
        Document(
            id="doc1",
            text="Machine learning is a subset of artificial intelligence that focuses on data and algorithms",
            metadata={"category": "AI", "source": "tutorial"}
        ),
        Document(
            id="doc2",
            text="Deep learning uses neural networks with multiple layers to learn complex patterns",
            metadata={"category": "AI", "source": "research"}
        ),
        Document(
            id="doc3",
            text="Natural language processing enables computers to understand and generate human language",
            metadata={"category": "NLP", "source": "overview"}
        ),
        Document(
            id="doc4",
            text="Vector databases store and retrieve high-dimensional embeddings efficiently using approximate nearest neighbor search",
            metadata={"category": "Database", "source": "technical"}
        ),
        Document(
            id="doc5",
            text="Python is a popular programming language for data science and machine learning applications",
            metadata={"category": "Programming", "source": "guide"}
        ),
        Document(
            id="doc6",
            text="HNSW is a graph-based algorithm for approximate nearest neighbor search with excellent recall and speed",
            metadata={"category": "Algorithms", "source": "research"}
        ),
        Document(
            id="doc7",
            text="Transformers revolutionized NLP by using self-attention mechanisms instead of recurrent architectures",
            metadata={"category": "AI", "source": "research"}
        ),
        Document(
            id="doc8",
            text="Quantization compresses vectors by reducing precision, achieving 4x to 64x compression ratios",
            metadata={"category": "Optimization", "source": "technical"}
        ),
        Document(
            id="doc9",
            text="Cosine similarity measures the angle between vectors and is commonly used for text embeddings",
            metadata={"category": "Math", "source": "tutorial"}
        ),
        Document(
            id="doc10",
            text="Kubernetes orchestrates containerized applications across clusters for scalable deployments",
            metadata={"category": "DevOps", "source": "guide"}
        ),
        Document(
            id="doc11",
            text="GPT models use transformer architecture with billions of parameters to generate coherent text",
            metadata={"category": "AI", "source": "research"}
        ),
        Document(
            id="doc12",
            text="Binary quantization reduces vectors to single bits per dimension for extreme compression",
            metadata={"category": "Optimization", "source": "technical"}
        ),
        Document(
            id="doc13",
            text="Locality sensitive hashing maps similar vectors to the same hash buckets for fast retrieval",
            metadata={"category": "Algorithms", "source": "research"}
        ),
        Document(
            id="doc14",
            text="Docker containers package applications with dependencies for consistent deployment across environments",
            metadata={"category": "DevOps", "source": "guide"}
        ),
        Document(
            id="doc15",
            text="BERT uses bidirectional transformers to learn contextual word representations for NLP tasks",
            metadata={"category": "NLP", "source": "research"}
        ),
    ]

    return documents


def build_indexes(
    vectors: List[Vector],
    metadata: List[Dict],
    dimension: int
) -> Dict[str, Tuple[object, float]]:
    """Build all indexes and measure build time.

    Returns:
        Dict mapping index name to (index, build_time)
    """
    indexes = {}

    # Linear Scan (baseline)
    print("\nBuilding Linear Scan index...")
    start = time.time()
    linear_index = LinearScan(dimension)
    linear_index.add(vectors, metadata)
    build_time = time.time() - start
    indexes["Linear Scan"] = (linear_index, build_time)
    print(f"  Built in {build_time:.4f}s")

    # KD-Tree
    print("\nBuilding KD-Tree index...")
    start = time.time()
    kdtree_index = KDTree(dimension)
    kdtree_index.add(vectors, metadata)
    build_time = time.time() - start
    indexes["KD-Tree"] = (kdtree_index, build_time)
    print(f"  Built in {build_time:.4f}s")

    # LSH
    print("\nBuilding LSH index...")
    start = time.time()
    lsh_index = LSH(dimension, num_tables=5, num_bits=10)
    lsh_index.add(vectors, metadata)
    build_time = time.time() - start
    indexes["LSH"] = (lsh_index, build_time)
    print(f"  Built in {build_time:.4f}s")

    # HNSW
    print("\nBuilding HNSW index...")
    start = time.time()
    hnsw_index = HNSW(dimension, M=16, ef_construction=200)
    hnsw_index.add(vectors, metadata)
    build_time = time.time() - start
    indexes["HNSW"] = (hnsw_index, build_time)
    print(f"  Built in {build_time:.4f}s")

    # IVF
    print("\nBuilding IVF index...")
    start = time.time()
    ivf_index = IVF(dimension, n_clusters=4, n_probe=2)  # Small n_clusters for small dataset
    ivf_index.add(vectors, metadata)
    build_time = time.time() - start
    indexes["IVF"] = (ivf_index, build_time)
    print(f"  Built in {build_time:.4f}s")

    return indexes


def run_queries(
    queries: List[Tuple[str, str]],
    embedder: SimpleEmbedder,
    indexes: Dict[str, Tuple[object, float]],
    k: int = 3
):
    """Run queries and compare results across indexes."""

    for query_id, query_text in queries:
        print("\n" + "=" * 80)
        print(f"Query: '{query_text}'")
        print("=" * 80)

        # Generate query embedding
        query_vector = embedder.embed(query_text)

        # Run on each index
        for index_name, (index, build_time) in indexes.items():
            print(f"\n{index_name}:")

            # Search
            start = time.time()
            result = index.search(query_vector, k=k)
            search_time = time.time() - start

            # Display results
            print(f"  Search time: {search_time*1000:.2f}ms")

            # Show distance computations if available
            if hasattr(result, 'comparisons'):
                print(f"  Distance computations: {result.comparisons}")
            elif hasattr(result, 'distance_computations'):
                print(f"  Distance computations: {result.distance_computations}")

            # Display results using indices, distances, metadata
            if result.indices:
                print(f"  Top {len(result.indices)} results:")
                for i, (idx, dist) in enumerate(zip(result.indices, result.distances), 1):
                    meta = result.metadata[i-1] if result.metadata else {}
                    doc_id = meta.get("id", f"idx_{idx}")
                    category = meta.get("category", "unknown")
                    text = meta.get("text", "")[:60] + "..."
                    print(f"    {i}. [{doc_id}] ({category}) dist={dist:.4f}")
                    print(f"       {text}")


def compare_performance(
    embedder: SimpleEmbedder,
    indexes: Dict[str, Tuple[object, float]],
    num_queries: int = 50
):
    """Benchmark query performance across indexes."""

    print("\n" + "=" * 80)
    print("Performance Benchmark")
    print("=" * 80)

    # Generate random queries
    query_texts = [
        "machine learning algorithms",
        "natural language processing",
        "vector similarity search",
        "neural network training",
        "data compression techniques",
    ]

    queries = [embedder.embed(text) for text in query_texts] * (num_queries // len(query_texts))

    results = {}

    for index_name, (index, build_time) in indexes.items():
        print(f"\nBenchmarking {index_name}...")

        times = []
        comparisons = []

        for query in queries:
            start = time.time()
            result = index.search(query, k=10)
            elapsed = time.time() - start

            times.append(elapsed)

            # Get comparisons if available
            if hasattr(result, 'comparisons'):
                comparisons.append(result.comparisons)
            elif hasattr(result, 'distance_computations'):
                comparisons.append(result.distance_computations)
            else:
                comparisons.append(0)  # Not tracked

        avg_time = np.mean(times) * 1000  # Convert to ms
        avg_comparisons = np.mean(comparisons)
        qps = 1.0 / np.mean(times)

        results[index_name] = {
            "build_time": build_time,
            "avg_query_time_ms": avg_time,
            "avg_comparisons": avg_comparisons,
            "qps": qps
        }

    # Display comparison table
    print("\n" + "=" * 80)
    print("Performance Summary")
    print("=" * 80)
    print(f"\n{'Index':<15} {'Build(s)':<12} {'Query(ms)':<12} {'Comparisons':<12} {'QPS':<10}")
    print("-" * 80)

    for index_name, metrics in results.items():
        print(f"{index_name:<15} "
              f"{metrics['build_time']:<12.4f} "
              f"{metrics['avg_query_time_ms']:<12.2f} "
              f"{metrics['avg_comparisons']:<12.0f} "
              f"{metrics['qps']:<10.0f}")

    return results


def main():
    """Run complete document demo."""
    print("=" * 80)
    print("Vector Database Document Demo")
    print("=" * 80)

    # Configuration
    dimension = 128
    k = 3

    # Step 1: Load documents
    print("\n[1/5] Loading documents...")
    documents = load_sample_documents()
    print(f"  Loaded {len(documents)} documents")

    # Step 2: Generate embeddings
    print("\n[2/5] Generating embeddings...")
    embedder = SimpleEmbedder(dimension)
    vectors = embedder.embed_batch([doc.text for doc in documents])
    print(f"  Generated {len(vectors)} embeddings of dimension {dimension}")

    # Prepare metadata
    metadata = []
    for doc, vec in zip(documents, vectors):
        meta = doc.metadata.copy()
        meta["id"] = doc.id
        meta["text"] = doc.text
        metadata.append(meta)

    # Step 3: Build indexes
    print("\n[3/5] Building indexes...")
    indexes = build_indexes(vectors, metadata, dimension)

    # Step 4: Run example queries
    print("\n[4/5] Running example queries...")
    queries = [
        ("q1", "neural networks and deep learning"),
        ("q2", "vector search algorithms"),
        ("q3", "compression and optimization"),
    ]
    run_queries(queries, embedder, indexes, k=k)

    # Step 5: Performance benchmark
    print("\n[5/5] Running performance benchmark...")
    perf_results = compare_performance(embedder, indexes, num_queries=50)

    # Final summary
    print("\n" + "=" * 80)
    print("Key Takeaways")
    print("=" * 80)
    print("""
1. Linear Scan: Slowest but 100% accurate (baseline)
2. KD-Tree: Fails at high dimensions (curse of dimensionality)
3. LSH: Fast but lower recall, good for very large datasets
4. HNSW: Best balance of speed and accuracy (state-of-the-art)
5. IVF: Good for billion-scale with quantization

Production Recommendations:
- Small datasets (<100K): HNSW
- Large datasets (>1M): IVF + Product Quantization
- Extremely large (>100M): HNSW + Scalar Quantization or IVF + PQ
- Real-time updates: HNSW (supports efficient inserts)
- Batch updates: IVF (rebuild clusters periodically)

Next Steps:
- Use real embeddings (sentence-transformers, OpenAI)
- Add metadata filtering
- Implement persistence (save/load indexes)
- Add API layer (FastAPI)
- Scale to larger datasets (millions of vectors)
    """)


if __name__ == "__main__":
    main()
