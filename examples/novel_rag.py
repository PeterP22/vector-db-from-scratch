"""RAG System for Your Novel.

This script loads a PDF novel, chunks it into passages, embeds them with Nvidia's
Nemotron model, and allows interactive Q&A about the book.

Installation:
    pip install torch transformers pypdf
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from typing import List, Dict, Tuple
import time
from dataclasses import dataclass
import re

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
    try:
        import PyPDF2 as pypdf
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
        """Initialize PDF loader.

        Args:
            chunk_size: Target number of words per chunk
            overlap: Number of words to overlap between chunks
        """
        self.chunk_size = chunk_size
        self.overlap = overlap

    def load(self, pdf_path: str) -> List[Chunk]:
        """Load PDF and split into chunks.

        Args:
            pdf_path: Path to PDF file

        Returns:
            List of text chunks with metadata
        """
        print(f"Loading PDF: {pdf_path}")

        # Read PDF
        chunks = []
        with open(pdf_path, 'rb') as f:
            pdf_reader = pypdf.PdfReader(f)
            num_pages = len(pdf_reader.pages)
            print(f"  Found {num_pages} pages")

            # Extract text from each page
            for page_num, page in enumerate(pdf_reader.pages):
                text = page.extract_text()

                # Clean text
                text = self._clean_text(text)

                if not text.strip():
                    continue

                # Split page into chunks
                page_chunks = self._chunk_text(text, page_num)
                chunks.extend(page_chunks)

        print(f"  Created {len(chunks)} chunks")
        return chunks

    def _clean_text(self, text: str) -> str:
        """Clean extracted text."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)

        # Remove page numbers and headers (common patterns)
        text = re.sub(r'\n\d+\n', '\n', text)

        # Fix common OCR issues
        text = text.replace('�', '')

        return text.strip()

    def _chunk_text(self, text: str, page_num: int) -> List[Chunk]:
        """Split text into overlapping chunks.

        Args:
            text: Text to chunk
            page_num: Page number (0-indexed)

        Returns:
            List of chunks
        """
        words = text.split()
        chunks = []
        chunk_index = 0

        i = 0
        while i < len(words):
            # Get chunk of words
            chunk_words = words[i:i + self.chunk_size]
            chunk_text = ' '.join(chunk_words)

            # Create chunk
            chunk = Chunk(
                id=f"page{page_num}_chunk{chunk_index}",
                text=chunk_text,
                page=page_num,
                chunk_index=chunk_index,
                metadata={
                    "page": page_num + 1,  # 1-indexed for display
                    "chunk_index": chunk_index,
                    "num_words": len(chunk_words)
                }
            )
            chunks.append(chunk)

            # Move forward with overlap
            i += self.chunk_size - self.overlap
            chunk_index += 1

        return chunks


class NvidiaEmbedder:
    """Nvidia Nemotron embedding model."""

    def __init__(
        self,
        model_name: str = "nvidia/llama-3.2-nv-embedqa-1b-v2",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """Initialize embedder."""
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
            test_embedding = test_output.last_hidden_state.mean(dim=1)
            self.dimension = test_embedding.shape[1]

        print(f"Model loaded. Embedding dimension: {self.dimension}")

    def embed(self, text: str) -> Vector:
        """Generate embedding for text."""
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: List[str], batch_size: int = 16) -> List[Vector]:
        """Generate embeddings for multiple texts.

        Args:
            texts: List of texts
            batch_size: Batch size (reduce if running out of memory)

        Returns:
            List of embeddings
        """
        all_embeddings = []

        print(f"Embedding {len(texts)} chunks...", end='', flush=True)

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
                embeddings = outputs.last_hidden_state.mean(dim=1)

                # Normalize
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

                # Convert to numpy
                embeddings = embeddings.cpu().numpy().astype(np.float32)
                all_embeddings.extend(embeddings)

                # Progress
                if (i + batch_size) % 100 == 0:
                    print(f".", end='', flush=True)

        print(" Done!")
        return all_embeddings


class NovelRAG:
    """RAG system for querying a novel."""

    def __init__(
        self,
        pdf_path: str,
        model_name: str = "nvidia/llama-3.2-nv-embedqa-1b-v2",
        chunk_size: int = 500,
        overlap: int = 50
    ):
        """Initialize Novel RAG system.

        Args:
            pdf_path: Path to PDF novel
            model_name: Nvidia embedding model to use
            chunk_size: Words per chunk
            overlap: Overlap between chunks
        """
        self.pdf_path = pdf_path

        # Load PDF
        print("\n" + "=" * 80)
        print("Step 1: Loading PDF")
        print("=" * 80)
        loader = PDFLoader(chunk_size=chunk_size, overlap=overlap)
        self.chunks = loader.load(pdf_path)

        # Initialize embedder
        print("\n" + "=" * 80)
        print("Step 2: Initializing Nvidia Embedder")
        print("=" * 80)
        self.embedder = NvidiaEmbedder(model_name=model_name)

        # Generate embeddings
        print("\n" + "=" * 80)
        print("Step 3: Generating Embeddings")
        print("=" * 80)
        start = time.time()
        chunk_texts = [chunk.text for chunk in self.chunks]
        self.vectors = self.embedder.embed_batch(chunk_texts, batch_size=16)
        embed_time = time.time() - start
        print(f"Generated {len(self.vectors)} embeddings in {embed_time:.1f}s")
        print(f"Average: {embed_time/len(self.vectors)*1000:.1f}ms per chunk")

        # Build index
        print("\n" + "=" * 80)
        print("Step 4: Building HNSW Index")
        print("=" * 80)
        start = time.time()
        self.index = HNSW(
            dimension=self.embedder.dimension,
            M=16,
            ef_construction=200
        )

        # Prepare metadata
        metadata = []
        for chunk in self.chunks:
            meta = chunk.metadata.copy()
            meta["id"] = chunk.id
            meta["text"] = chunk.text
            metadata.append(meta)

        self.index.add(self.vectors, metadata)
        build_time = time.time() - start
        print(f"Built HNSW index in {build_time:.2f}s")

        print("\n" + "=" * 80)
        print("RAG System Ready!")
        print("=" * 80)
        print(f"Indexed {len(self.chunks)} chunks from your novel")
        print(f"Ready to answer questions!")

    def query(self, question: str, k: int = 5) -> List[Tuple[str, float, Dict]]:
        """Query the novel.

        Args:
            question: Question to ask
            k: Number of relevant passages to retrieve

        Returns:
            List of (text, similarity, metadata) tuples
        """
        # Embed query
        query_vector = self.embedder.embed(question)

        # Search
        result = self.index.search(query_vector, k=k)

        # Format results
        results = []
        for i, (idx, dist) in enumerate(zip(result.indices, result.distances)):
            meta = result.metadata[i]  # Use position in results, not original index
            similarity = 1 - dist  # Convert distance to similarity
            results.append((meta["text"], similarity, meta))

        return results

    def interactive(self):
        """Interactive Q&A session."""
        print("\n" + "=" * 80)
        print("Interactive Q&A Mode")
        print("=" * 80)
        print("Ask questions about your novel. Type 'quit' to exit.\n")

        while True:
            # Get question
            question = input("Question: ").strip()

            if not question:
                continue

            if question.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye!")
                break

            # Query
            print("\nSearching...")
            start = time.time()
            results = self.query(question, k=3)
            search_time = time.time() - start

            # Display results
            print(f"\nTop 3 relevant passages (found in {search_time*1000:.1f}ms):")
            print("=" * 80)

            for i, (text, similarity, meta) in enumerate(results, 1):
                print(f"\n{i}. Page {meta['page']}, Chunk {meta['chunk_index']} (Similarity: {similarity:.3f})")
                print("-" * 80)

                # Show text with word limit
                words = text.split()
                if len(words) > 150:
                    display_text = ' '.join(words[:150]) + "..."
                else:
                    display_text = text

                print(display_text)

            print("\n" + "=" * 80 + "\n")


def main():
    """Run novel RAG demo."""
    print("=" * 80)
    print("Novel RAG System - Query Your Book with AI")
    print("=" * 80)

    # Configuration
    pdf_path = "/Users/peterpreketes/vector-db-from-scratch/3%man.pdf"

    # Check file exists
    if not os.path.exists(pdf_path):
        print(f"\nError: PDF not found at {pdf_path}")
        print("Please update the pdf_path variable in the script.")
        return

    # Initialize RAG system
    try:
        rag = NovelRAG(
            pdf_path=pdf_path,
            model_name="nvidia/llama-3.2-nv-embedqa-1b-v2",
            chunk_size=500,  # 500 words per chunk
            overlap=50       # 50 word overlap
        )
    except Exception as e:
        print(f"\nError initializing RAG system: {e}")
        print("\nMake sure you have:")
        print("1. Installed dependencies: pip install torch transformers pypdf")
        print("2. Logged in to Hugging Face: huggingface-cli login")
        print("3. Accepted the model license at:")
        print("   https://huggingface.co/nvidia/llama-3.2-nv-embedqa-1b-v2")
        return

    # Start interactive mode
    rag.interactive()


if __name__ == "__main__":
    main()
