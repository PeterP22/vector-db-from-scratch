"""Example 3: RAG (Retrieval-Augmented Generation) - Retrieval Only

What you'll learn:
- Building a complete RAG retrieval pipeline
- PDF document loading and intelligent chunking
- Embedding caching (instant reload on second run)
- Interactive Q&A with retrieved passages
- HNSW index for fast semantic search

RAG Components (this example):
✅ Document loading (PDF → text chunks)
✅ Embedding generation (with caching)
✅ Vector search (retrieve top-K passages)
❌ LLM synthesis (see next example for this!)

Key Concepts:
- Document chunking: Breaking long text into searchable pieces
- Overlap strategy: Avoiding context loss at chunk boundaries
- Smart caching: Save embeddings to disk (huge time saver!)
- Top-K retrieval: Get most relevant passages for a query

Features:
- GPU acceleration (Mac MPS or NVIDIA CUDA)
- Persistent caching (~2s load vs 50s regeneration)
- Interactive Q&A mode

Prerequisites:
    pip install torch transformers pypdf

Previous: 2_nvidia_embeddings_demo.py (production embeddings)
Next: 4_complete_rag_system.py (add LLM for answer synthesis!)
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from typing import List, Dict, Tuple
import time
from dataclasses import dataclass
import re
import pickle
from pathlib import Path

from src.core.vector import Vector
from src.indexes.hnsw import HNSW

try:
    from transformers import AutoTokenizer, AutoModel
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Error: transformers not installed. Install with: pip install torch transformers")
    sys.exit(1)

try:
    import pypdf
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    print("Error: pypdf not installed. Install with: pip install pypdf")
    sys.exit(1)


@dataclass
class Chunk:
    """A chunk of text from the novel."""
    id: str
    text: str
    page: int
    chunk_index: int
    metadata: Dict


class PDFLoader:
    """Load and chunk PDF documents."""

    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        """Initialize PDF loader."""
        self.chunk_size = chunk_size
        self.overlap = overlap

    def load(self, pdf_path: str) -> List[Chunk]:
        """Load PDF and split into chunks."""
        print(f"Loading PDF: {pdf_path}")

        chunks = []
        with open(pdf_path, 'rb') as f:
            pdf_reader = pypdf.PdfReader(f)
            num_pages = len(pdf_reader.pages)
            print(f"  Found {num_pages} pages")

            for page_num, page in enumerate(pdf_reader.pages):
                text = page.extract_text()
                text = self._clean_text(text)

                if not text.strip():
                    continue

                page_chunks = self._chunk_text(text, page_num)
                chunks.extend(page_chunks)

        print(f"  Created {len(chunks)} chunks")
        return chunks

    def _clean_text(self, text: str) -> str:
        """Clean extracted text."""
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n\d+\n', '\n', text)
        text = text.replace('�', '')
        return text.strip()

    def _chunk_text(self, text: str, page_num: int) -> List[Chunk]:
        """Split text into overlapping chunks."""
        words = text.split()
        chunks = []
        chunk_index = 0

        i = 0
        while i < len(words):
            chunk_words = words[i:i + self.chunk_size]
            chunk_text = ' '.join(chunk_words)

            chunk = Chunk(
                id=f"page{page_num}_chunk{chunk_index}",
                text=chunk_text,
                page=page_num,
                chunk_index=chunk_index,
                metadata={
                    "page": page_num + 1,
                    "chunk_index": chunk_index,
                    "num_words": len(chunk_words)
                }
            )
            chunks.append(chunk)

            i += self.chunk_size - self.overlap
            chunk_index += 1

        return chunks


class NvidiaEmbedder:
    """Nvidia Nemotron embedding model with GPU support."""

    def __init__(
        self,
        model_name: str = "nvidia/llama-3.2-nv-embedqa-1b-v2",
        use_gpu: bool = True
    ):
        """Initialize embedder with GPU support.

        Args:
            model_name: Hugging Face model name
            use_gpu: Whether to use GPU (Mac uses MPS, otherwise CUDA)
        """
        # Detect best device
        if use_gpu:
            if torch.backends.mps.is_available():
                device = "mps"  # Apple Silicon GPU
                print("🚀 Using Mac GPU (Metal Performance Shaders)")
            elif torch.cuda.is_available():
                device = "cuda"  # NVIDIA GPU
                print("🚀 Using NVIDIA GPU")
            else:
                device = "cpu"
                print("⚠️  No GPU detected, using CPU (will be slower)")
        else:
            device = "cpu"
            print("Using CPU")

        print(f"Loading {model_name}...")
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
            test_embedding = test_output.last_hidden_state.mean(dim=1)
            self.dimension = test_embedding.shape[1]

        print(f"✓ Model loaded. Embedding dimension: {self.dimension}")

    def embed(self, text: str) -> Vector:
        """Generate embedding for text."""
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: List[str], batch_size: int = 32) -> List[Vector]:
        """Generate embeddings for multiple texts."""
        all_embeddings = []

        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]

                inputs = self.tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                outputs = self.model(**inputs)
                embeddings = outputs.last_hidden_state.mean(dim=1)
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                embeddings = embeddings.cpu().numpy().astype(np.float32)
                all_embeddings.extend(embeddings)

                # Progress indicator
                if (i + batch_size) % 50 == 0:
                    progress = min(i + batch_size, len(texts))
                    print(f"  Progress: {progress}/{len(texts)} chunks embedded")

        return all_embeddings


class NovelRAG:
    """RAG system with caching support."""

    def __init__(
        self,
        pdf_path: str,
        model_name: str = "nvidia/llama-3.2-nv-embedqa-1b-v2",
        chunk_size: int = 500,
        overlap: int = 50,
        cache_dir: str = ".cache",
        use_gpu: bool = True
    ):
        """Initialize Novel RAG with caching.

        Args:
            pdf_path: Path to PDF
            model_name: Embedding model
            chunk_size: Words per chunk
            overlap: Overlap between chunks
            cache_dir: Directory to cache embeddings
            use_gpu: Whether to use GPU
        """
        self.pdf_path = pdf_path
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

        # Create cache filename based on PDF and settings
        pdf_name = Path(pdf_path).stem
        cache_filename = f"{pdf_name}_chunks{chunk_size}_overlap{overlap}.pkl"
        self.cache_path = self.cache_dir / cache_filename

        # Try to load from cache
        if self.cache_path.exists():
            print("\n" + "=" * 80)
            print("📦 Loading from cache (much faster!)")
            print("=" * 80)
            print(f"Cache file: {self.cache_path}")

            with open(self.cache_path, 'rb') as f:
                cache_data = pickle.load(f)

            self.chunks = cache_data['chunks']
            self.vectors = cache_data['vectors']
            embedding_dim = cache_data['embedding_dim']

            print(f"✓ Loaded {len(self.chunks)} chunks from cache")
            print(f"✓ Loaded {len(self.vectors)} embeddings (dim={embedding_dim})")

            # Initialize embedder just for queries (model still needed)
            print("\n" + "=" * 80)
            print("Loading Model for Queries")
            print("=" * 80)
            self.embedder = NvidiaEmbedder(model_name=model_name, use_gpu=use_gpu)

        else:
            print("\n" + "=" * 80)
            print("🆕 First time setup - will cache for next time")
            print("=" * 80)

            # Load PDF
            print("\nStep 1: Loading PDF")
            print("-" * 80)
            loader = PDFLoader(chunk_size=chunk_size, overlap=overlap)
            self.chunks = loader.load(pdf_path)

            # Initialize embedder
            print("\nStep 2: Initializing Model")
            print("-" * 80)
            self.embedder = NvidiaEmbedder(model_name=model_name, use_gpu=use_gpu)

            # Generate embeddings
            print("\nStep 3: Generating Embeddings")
            print("-" * 80)
            start = time.time()
            chunk_texts = [chunk.text for chunk in self.chunks]
            self.vectors = self.embedder.embed_batch(chunk_texts, batch_size=32)
            embed_time = time.time() - start

            print(f"✓ Generated {len(self.vectors)} embeddings in {embed_time:.1f}s")
            print(f"  Average: {embed_time/len(self.vectors)*1000:.0f}ms per chunk")

            # Save to cache
            print("\nStep 4: Saving to Cache")
            print("-" * 80)
            cache_data = {
                'chunks': self.chunks,
                'vectors': self.vectors,
                'embedding_dim': self.embedder.dimension,
                'model_name': model_name,
                'chunk_size': chunk_size,
                'overlap': overlap
            }

            with open(self.cache_path, 'wb') as f:
                pickle.dump(cache_data, f)

            print(f"✓ Cached to: {self.cache_path}")
            print(f"  Next time will be instant!")

        # Build index
        print("\nBuilding HNSW Index")
        print("-" * 80)
        start = time.time()
        self.index = HNSW(
            dimension=self.embedder.dimension,
            M=16,
            ef_construction=200
        )

        metadata = []
        for chunk in self.chunks:
            meta = chunk.metadata.copy()
            meta["id"] = chunk.id
            meta["text"] = chunk.text
            metadata.append(meta)

        self.index.add(self.vectors, metadata)
        build_time = time.time() - start
        print(f"✓ Built HNSW index in {build_time:.2f}s")

        print("\n" + "=" * 80)
        print("✅ RAG System Ready!")
        print("=" * 80)
        print(f"📚 Indexed {len(self.chunks)} chunks from your novel")
        print(f"⚡ Ready to answer questions!")

    def query(self, question: str, k: int = 5) -> List[Tuple[str, float, Dict]]:
        """Query the novel."""
        # Embed query
        query_vector = self.embedder.embed(question)

        # Search
        result = self.index.search(query_vector, k=k)

        # Format results
        results = []
        for i, (idx, dist) in enumerate(zip(result.indices, result.distances)):
            meta = result.metadata[i]
            similarity = 1 - dist
            results.append((meta["text"], similarity, meta))

        return results

    def interactive(self):
        """Interactive Q&A session."""
        print("\n" + "=" * 80)
        print("💬 Interactive Q&A Mode")
        print("=" * 80)
        print("Ask questions about your novel. Type 'quit' or 'exit' to stop.\n")

        question_count = 0

        while True:
            # Get question
            try:
                question = input("\n📖 Question: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n\nGoodbye!")
                break

            if not question:
                continue

            if question.lower() in ['quit', 'exit', 'q', 'bye']:
                print("\nGoodbye! 👋")
                break

            # Query
            question_count += 1
            start = time.time()
            results = self.query(question, k=3)
            search_time = time.time() - start

            # Display results
            print(f"\n{'='*80}")
            print(f"Top 3 Relevant Passages (found in {search_time*1000:.0f}ms)")
            print(f"{'='*80}")

            for i, (text, similarity, meta) in enumerate(results, 1):
                print(f"\n{i}. 📄 Page {meta['page']} (Similarity: {similarity:.3f})")
                print("-" * 80)

                # Show text with word limit
                words = text.split()
                if len(words) > 150:
                    display_text = ' '.join(words[:150]) + "..."
                else:
                    display_text = text

                print(display_text)

        print(f"\n✅ Answered {question_count} questions!")


def main():
    """Run interactive novel RAG."""
    print("=" * 80)
    print("📚 Interactive Novel RAG with GPU & Caching")
    print("=" * 80)

    # Configuration
    pdf_path = "/Users/peterpreketes/vector-db-from-scratch/3%man.pdf"

    if not os.path.exists(pdf_path):
        print(f"\n❌ Error: PDF not found at {pdf_path}")
        return

    # Initialize RAG system
    try:
        rag = NovelRAG(
            pdf_path=pdf_path,
            model_name="nvidia/llama-3.2-nv-embedqa-1b-v2",
            chunk_size=500,
            overlap=50,
            cache_dir=".cache",
            use_gpu=True  # Use Mac GPU (MPS) or NVIDIA GPU
        )
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure you have:")
        print("1. pip install torch transformers pypdf")
        print("2. huggingface-cli login")
        print("3. Accepted model license at:")
        print("   https://huggingface.co/nvidia/llama-3.2-nv-embedqa-1b-v2")
        return

    # Start interactive mode
    rag.interactive()


if __name__ == "__main__":
    main()
