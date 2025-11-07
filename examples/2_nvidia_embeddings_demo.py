"""Example 2: Production-Quality Embeddings with Nvidia Nemotron

What you'll learn:
- Moving from simple embeddings to state-of-the-art models
- Using Nvidia's Nemotron models (1B and 8B parameter versions)
- GPU acceleration (Mac MPS or NVIDIA CUDA)
- Comparing different embedding model sizes

Models demonstrated:
- llama-3.2-nv-embedqa-1b-v2: Fast, lightweight (2048 dims)
- llama-embed-nemotron-8b: Best quality (4096 dims)

Key Concepts:
- Transformer models: BERT/LLaMA-based embeddings
- GPU acceleration: 4x faster than CPU
- Model size trade-offs: Speed vs quality

Prerequisites:
    pip install torch transformers

Previous: 1_basic_index_comparison.py (understand the basics first)
Next: 3_rag_retrieval_only.py (build a RAG system)
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from typing import List, Dict, Tuple
import time
from dataclasses import dataclass

from src.core.vector import Vector
from src.indexes.hnsw import HNSW
from src.indexes.ivf import IVF

try:
    from transformers import AutoTokenizer, AutoModel
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers not installed. Install with: pip install torch transformers")


@dataclass
class Document:
    """Document with text and metadata."""
    id: str
    text: str
    metadata: Dict


class NvidiaEmbedder:
    """Wrapper for Nvidia Nemotron embedding models.

    Supports:
    - nvidia/llama-embed-nemotron-8b (8B params, best quality)
    - nvidia/llama-3.2-nv-embedqa-1b-v2 (1B params, lightweight)
    """

    def __init__(
        self,
        model_name: str = "nvidia/llama-3.2-nv-embedqa-1b-v2",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """Initialize Nvidia embedder.

        Args:
            model_name: Hugging Face model name
                - "nvidia/llama-embed-nemotron-8b" (best quality, slower)
                - "nvidia/llama-3.2-nv-embedqa-1b-v2" (fast, good for QA)
            device: Device to use ('cuda' or 'cpu')
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers not installed. Install with: pip install torch transformers")

        print(f"Loading {model_name} on {device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.model = self.model.to(device)
        self.model.eval()
        self.device = device
        self.model_name = model_name

        # Get embedding dimension
        with torch.no_grad():
            test_input = self.tokenizer("test", return_tensors="pt", padding=True, truncation=True)
            test_input = {k: v.to(self.device) for k, v in test_input.items()}
            test_output = self.model(**test_input)
            # Mean pooling
            test_embedding = test_output.last_hidden_state.mean(dim=1)
            self.dimension = test_embedding.shape[1]

        print(f"Model loaded. Embedding dimension: {self.dimension}")

    def embed(self, text: str) -> Vector:
        """Generate embedding for a single text."""
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: List[str], batch_size: int = 32) -> List[Vector]:
        """Generate embeddings for multiple texts with batching.

        Args:
            texts: List of texts to embed
            batch_size: Batch size for processing

        Returns:
            List of embedding vectors
        """
        all_embeddings = []

        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]

                # Tokenize
                inputs = self.tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                # Get embeddings
                outputs = self.model(**inputs)

                # Mean pooling over sequence length
                embeddings = outputs.last_hidden_state.mean(dim=1)

                # Normalize
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

                # Convert to numpy
                embeddings = embeddings.cpu().numpy().astype(np.float32)
                all_embeddings.extend(embeddings)

        return all_embeddings


class NvidiaReranker:
    """Wrapper for Nvidia Nemotron reranking model.

    Uses nvidia/llama-3.2-nv-rerankqa-1b-v2 to rerank retrieved documents.
    """

    def __init__(
        self,
        model_name: str = "nvidia/llama-3.2-nv-rerankqa-1b-v2",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """Initialize Nvidia reranker.

        Args:
            model_name: Hugging Face model name
            device: Device to use ('cuda' or 'cpu')
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers not installed. Install with: pip install torch transformers")

        print(f"Loading reranker {model_name} on {device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.model = self.model.to(device)
        self.model.eval()
        self.device = device
        print("Reranker loaded.")

    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: int = None
    ) -> List[Tuple[int, float]]:
        """Rerank documents by relevance to query.

        Args:
            query: Query text
            documents: List of document texts
            top_k: Number of top documents to return (None = all)

        Returns:
            List of (document_index, score) tuples, sorted by score descending
        """
        if not documents:
            return []

        # Create query-document pairs
        pairs = [[query, doc] for doc in documents]

        with torch.no_grad():
            # Tokenize pairs
            inputs = self.tokenizer(
                pairs,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Get relevance scores
            outputs = self.model(**inputs)
            scores = outputs.logits.squeeze().cpu().numpy()

            # Handle single document case
            if len(documents) == 1:
                scores = [float(scores)]
            else:
                scores = scores.tolist()

        # Sort by score descending
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)

        if top_k:
            ranked = ranked[:top_k]

        return ranked


def load_sample_documents() -> List[Document]:
    """Load sample documents about AI and ML topics."""
    documents = [
        Document(
            id="doc1",
            text="Vector databases enable efficient similarity search over high-dimensional embeddings using approximate nearest neighbor algorithms like HNSW and IVF.",
            metadata={"category": "Vector DB", "topic": "Search"}
        ),
        Document(
            id="doc2",
            text="HNSW (Hierarchical Navigable Small World) builds a multi-layer graph structure that allows logarithmic time search complexity with excellent recall.",
            metadata={"category": "Algorithms", "topic": "HNSW"}
        ),
        Document(
            id="doc3",
            text="Product quantization compresses vectors by splitting them into subspaces and clustering each independently, achieving 16-64x compression ratios.",
            metadata={"category": "Compression", "topic": "Quantization"}
        ),
        Document(
            id="doc4",
            text="Retrieval-augmented generation (RAG) combines neural retrieval systems with large language models to generate factual, grounded responses.",
            metadata={"category": "RAG", "topic": "Generation"}
        ),
        Document(
            id="doc5",
            text="Embedding models like BERT, Sentence-BERT, and Contriever transform text into dense vector representations that capture semantic meaning.",
            metadata={"category": "Embeddings", "topic": "Models"}
        ),
        Document(
            id="doc6",
            text="Reranking models improve retrieval quality by scoring query-document pairs with cross-attention, trading speed for accuracy.",
            metadata={"category": "RAG", "topic": "Reranking"}
        ),
        Document(
            id="doc7",
            text="IVF (Inverted File Index) uses k-means clustering to partition vectors into Voronoi cells, enabling sublinear search with multi-probe strategies.",
            metadata={"category": "Algorithms", "topic": "IVF"}
        ),
        Document(
            id="doc8",
            text="SIMD optimizations enable processing 32-64 uint8 values simultaneously using CPU instructions like AVX2 and AVX-512.",
            metadata={"category": "Optimization", "topic": "SIMD"}
        ),
        Document(
            id="doc9",
            text="Approximate nearest neighbor search trades perfect recall for speed, achieving 95-99% accuracy while being 10-100x faster than exact search.",
            metadata={"category": "Search", "topic": "ANN"}
        ),
        Document(
            id="doc10",
            text="Semantic search uses embedding models to find documents by meaning rather than keyword matching, enabling conceptual queries.",
            metadata={"category": "Search", "topic": "Semantic"}
        ),
    ]
    return documents


def main():
    """Run Nvidia RAG demo."""
    print("=" * 80)
    print("Nvidia Nemotron RAG Integration Demo")
    print("=" * 80)

    if not TRANSFORMERS_AVAILABLE:
        print("\nError: transformers not installed!")
        print("Install with: pip install torch transformers")
        return

    # Check CUDA availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")
    if device == "cpu":
        print("Warning: Running on CPU. This will be slow. Consider using GPU for better performance.")

    # Step 1: Load documents
    print("\n[1/6] Loading documents...")
    documents = load_sample_documents()
    print(f"  Loaded {len(documents)} documents")

    # Step 2: Initialize embedder
    print("\n[2/6] Initializing Nvidia embedder...")
    print("  Using: nvidia/llama-3.2-nv-embedqa-1b-v2 (1B, optimized for QA)")
    print("  Note: Change to 'nvidia/llama-embed-nemotron-8b' for best quality")

    try:
        embedder = NvidiaEmbedder(
            model_name="nvidia/llama-3.2-nv-embedqa-1b-v2",
            device=device
        )
    except Exception as e:
        print(f"\nError loading model: {e}")
        print("\nMake sure you have:")
        print("1. Accepted the model license on Hugging Face")
        print("2. Logged in with: huggingface-cli login")
        return

    # Step 3: Generate embeddings
    print("\n[3/6] Generating embeddings...")
    start = time.time()
    vectors = embedder.embed_batch([doc.text for doc in documents])
    embed_time = time.time() - start
    print(f"  Generated {len(vectors)} embeddings in {embed_time:.2f}s")
    print(f"  Embedding dimension: {embedder.dimension}")

    # Prepare metadata
    metadata = []
    for doc, vec in zip(documents, vectors):
        meta = doc.metadata.copy()
        meta["id"] = doc.id
        meta["text"] = doc.text
        metadata.append(meta)

    # Step 4: Build index
    print("\n[4/6] Building HNSW index...")
    start = time.time()
    index = HNSW(dimension=embedder.dimension, M=16, ef_construction=200)
    index.add(vectors, metadata)
    build_time = time.time() - start
    print(f"  Built index in {build_time:.4f}s")

    # Step 5: Run queries
    print("\n[5/6] Running queries...")
    queries = [
        "How do vector databases enable fast similarity search?",
        "What compression techniques reduce memory usage?",
        "Explain how RAG systems work",
    ]

    for query_text in queries:
        print(f"\n{'='*80}")
        print(f"Query: '{query_text}'")
        print(f"{'='*80}")

        # Embed query
        query_vector = embedder.embed(query_text)

        # Search
        k = 3
        start = time.time()
        result = index.search(query_vector, k=k)
        search_time = time.time() - start

        print(f"\nTop {k} results (search time: {search_time*1000:.2f}ms):")
        for i, (idx, dist) in enumerate(zip(result.indices, result.distances), 1):
            meta = result.metadata[i-1] if result.metadata else {}
            doc_id = meta.get("id", f"idx_{idx}")
            category = meta.get("category", "unknown")
            text = meta.get("text", "")[:80] + "..."
            similarity = 1 - dist  # Convert distance to similarity
            print(f"\n  {i}. [{doc_id}] ({category}) - Similarity: {similarity:.4f}")
            print(f"     {text}")

    # Step 6: Optional reranking demo
    print(f"\n{'='*80}")
    print("[6/6] Reranking Demo (Optional)")
    print(f"{'='*80}")
    print("\nReranking uses nvidia/llama-3.2-nv-rerankqa-1b-v2 to improve results.")
    print("This is slower but more accurate than pure vector search.")
    print("\nSkipping reranking in this demo (requires additional model download).")
    print("To enable reranking, uncomment the reranker code below.")

    # Uncomment to enable reranking:
    # try:
    #     reranker = NvidiaReranker(device=device)
    #
    #     query_text = queries[0]
    #     query_vector = embedder.embed(query_text)
    #     result = index.search(query_vector, k=10)  # Get more candidates
    #
    #     # Extract document texts
    #     retrieved_docs = [result.metadata[i]["text"] for i in range(len(result.indices))]
    #
    #     # Rerank
    #     reranked = reranker.rerank(query_text, retrieved_docs, top_k=3)
    #
    #     print("\nReranked results:")
    #     for i, (doc_idx, score) in enumerate(reranked, 1):
    #         meta = result.metadata[doc_idx]
    #         print(f"\n  {i}. [{meta['id']}] - Score: {score:.4f}")
    #         print(f"     {meta['text'][:80]}...")
    # except Exception as e:
    #     print(f"Reranking error: {e}")

    # Summary
    print(f"\n{'='*80}")
    print("Summary")
    print(f"{'='*80}")
    print(f"\nEmbedding model: {embedder.model_name}")
    print(f"Embedding dimension: {embedder.dimension}")
    print(f"Index type: HNSW")
    print(f"Documents indexed: {len(documents)}")
    print(f"Average embedding time: {embed_time/len(documents)*1000:.2f}ms per doc")
    print(f"Index build time: {build_time:.4f}s")
    print(f"\nNext steps:")
    print("1. Try nvidia/llama-embed-nemotron-8b for better quality")
    print("2. Enable reranking for improved accuracy")
    print("3. Scale to larger document collections")
    print("4. Add metadata filtering")
    print("5. Integrate with your own documents")


if __name__ == "__main__":
    main()
