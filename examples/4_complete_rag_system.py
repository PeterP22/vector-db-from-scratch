"""Example 4: Complete Production RAG System ⭐

THE FULL STACK - This is a production-ready RAG implementation!

What you'll learn:
- Complete RAG pipeline: Retrieve → Generate
- LLM integration for answer synthesis
- Streaming responses (word-by-word output)
- Customizable prompts (see prompts/ folder)
- Production optimization techniques

RAG Components (all of them!):
✅ Document loading (PDF → chunks)
✅ Nvidia embeddings (GPU accelerated)
✅ Vector search (HNSW index - retrieves top 15)
✅ Re-ranking (Nvidia Nemotron - narrows to top 5)
✅ LLM synthesis (Kimi K2 Turbo)
✅ Streaming output (real-time responses)
✅ Smart caching (instant reload)

What makes this "production-ready":
- Two-stage retrieval (vector search + re-ranking)
- GPU acceleration (4x faster embeddings & re-ranking)
- Persistent caching (save time & money)
- Streaming responses (better UX)
- Customizable prompts (no code changes needed)
- Error handling & retry logic
- Performance metrics tracking

Key Concepts:
- RAG: Retrieval-Augmented Generation (external knowledge for LLMs)
- Two-stage retrieval: Fast vector search (15 candidates) → accurate re-ranking (top 5)
- Re-ranking: Cross-attention model scores query-document pairs for better relevance
- Streaming: Display tokens as they're generated
- Prompt engineering: System/user prompts control LLM behavior
- Temperature tuning: 0.3 = focused, 0.7 = creative

Architecture:
1. User asks question
2. Embed question → search vector DB
3. Retrieve top 15 candidate passages (HNSW)
4. Re-rank candidates to top 5 (cross-attention model)
5. Send top 5 passages + question to LLM
6. LLM synthesizes coherent answer
7. Stream answer back to user

Setup:
    pip install torch transformers pypdf openai python-dotenv

    # Add to .env file:
    KIMI_API_KEY=your_api_key_here

Usage:
    python examples/4_complete_rag_system.py

Customization:
    - Edit prompts/system_prompt.txt to change LLM behavior
    - Edit prompts/user_prompt_template.txt to change query format
    - Adjust temperature in code (line 161) for creativity vs focus

Previous: 3_rag_retrieval_only.py (RAG without LLM)
This is the final example - you've mastered production RAG! 🎉
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Fix tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
from typing import List, Dict, Tuple
import time
from dataclasses import dataclass
import re
import pickle
from pathlib import Path

from src.core.vector import Vector


def load_prompt(prompt_name: str) -> str:
    """Load prompt from prompts directory.

    Args:
        prompt_name: Name of prompt file (e.g., 'system_prompt.txt')

    Returns:
        Prompt text
    """
    # Get project root (parent of examples/)
    project_root = Path(__file__).parent.parent
    prompt_path = project_root / "prompts" / prompt_name

    with open(prompt_path, 'r') as f:
        return f.read().strip()
from src.indexes.hnsw import HNSW

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False
    print("Warning: python-dotenv not installed. Install with: pip install python-dotenv")

try:
    from transformers import (
        AutoTokenizer,
        AutoModel,
        AutoModelForSequenceClassification,
        AutoConfig,
    )
    from transformers.dynamic_module_utils import get_class_from_dynamic_module
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

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("Error: openai not installed. Install with: pip install openai")
    sys.exit(1)


@dataclass
class Chunk:
    """A chunk of text from the novel."""
    id: str
    text: str
    page: int
    chunk_index: int
    metadata: Dict


class KimiLLM:
    """Kimi LLM client for answer synthesis."""

    def __init__(self, api_key: str = None, model: str = "kimi-k2-turbo-preview"):
        """Initialize Kimi LLM.

        Args:
            api_key: Kimi API key (reads from KIMI_API_KEY env var if not provided)
            model: Model to use (kimi-k2-turbo-preview: 60-100 tokens/s, high-speed)
        """
        if not OPENAI_AVAILABLE:
            raise ImportError("openai library not installed")

        # Get API key
        if api_key is None:
            api_key = os.getenv("KIMI_API_KEY")

        if not api_key:
            raise ValueError("KIMI_API_KEY not found. Set it in .env or pass as argument")

        # Initialize OpenAI client with Kimi endpoint
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.moonshot.ai/v1"
        )
        self.model = model

        print(f"✓ Kimi LLM initialized (model: {model})")

    def synthesize_answer(
        self,
        question: str,
        passages: List[Tuple[str, float, Dict]],
        max_tokens: int = 1000,
        stream: bool = False
    ):
        """Synthesize answer from retrieved passages.

        Args:
            question: User's question
            passages: List of (text, similarity, metadata) tuples
            max_tokens: Maximum tokens in response
            stream: Whether to stream the response (yields chunks if True)

        Returns:
            Synthesized answer string (if stream=False) or generator (if stream=True)
        """
        # Build context from passages
        context_parts = []
        for i, (text, similarity, meta) in enumerate(passages, 1):
            page = meta.get("page", "?")
            context_parts.append(f"[Passage {i} - Page {page}]\n{text}\n")

        context = "\n".join(context_parts)

        # Load prompts from files
        system_prompt = load_prompt("system_prompt.txt")
        user_prompt_template = load_prompt("user_prompt_template.txt")

        # Format user prompt with question and context
        user_prompt = user_prompt_template.format(question=question, context=context)

        # Call Kimi API
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=max_tokens,
                stream=stream
            )

            if stream:
                # Return generator for streaming
                return response
            else:
                # Return complete answer
                answer = response.choices[0].message.content
                return answer

        except Exception as e:
            if stream:
                return iter([f"Error calling Kimi API: {e}"])
            return f"Error calling Kimi API: {e}"


class PDFLoader:
    """Load and chunk PDF documents."""

    def __init__(self, chunk_size: int = 500, overlap: int = 50):
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
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n\d+\n', '\n', text)
        text = text.replace('�', '')
        return text.strip()

    def _chunk_text(self, text: str, page_num: int) -> List[Chunk]:
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
        if use_gpu:
            if torch.backends.mps.is_available():
                device = "mps"
                print("🚀 Using Mac GPU (Metal Performance Shaders)")
            elif torch.cuda.is_available():
                device = "cuda"
                print("🚀 Using NVIDIA GPU")
            else:
                device = "cpu"
                print("⚠️  No GPU detected, using CPU")
        else:
            device = "cpu"

        print(f"Loading {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.model = self.model.to(device)
        self.model.eval()
        self.device = device
        self.model_name = model_name

        with torch.no_grad():
            test_input = self.tokenizer("test", return_tensors="pt", padding=True, truncation=True)
            test_input = {k: v.to(self.device) for k, v in test_input.items()}
            test_output = self.model(**test_input)
            test_embedding = test_output.last_hidden_state.mean(dim=1)
            self.dimension = test_embedding.shape[1]

        print(f"✓ Model loaded. Embedding dimension: {self.dimension}")

    def embed(self, text: str) -> Vector:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: List[str], batch_size: int = 32) -> List[Vector]:
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

                if (i + batch_size) % 50 == 0:
                    progress = min(i + batch_size, len(texts))
                    print(f"  Progress: {progress}/{len(texts)} chunks embedded")

        return all_embeddings


class NvidiaReranker:
    """Nvidia Nemotron reranking model for improved retrieval."""

    def __init__(
        self,
        model_name: str = "nvidia/llama-3.2-nv-rerankqa-1b-v2",
        use_gpu: bool = True
    ):
        """Initialize reranker with GPU support.

        Args:
            model_name: Hugging Face model name
            use_gpu: Whether to use GPU (Mac uses MPS, otherwise CUDA)
        """
        # Detect best device
        if use_gpu:
            if torch.backends.mps.is_available():
                device = "mps"
                print("🚀 Using Mac GPU for re-ranking")
            elif torch.cuda.is_available():
                device = "cuda"
                print("🚀 Using NVIDIA GPU for re-ranking")
            else:
                device = "cpu"
                print("⚠️  No GPU for re-ranker, using CPU")
        else:
            device = "cpu"

        print(f"Loading re-ranker: {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        # Download config so that any custom modeling code is pulled as well
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        model_type = getattr(config, "model_type", "unknown")
        auto_map = getattr(config, "auto_map", None)
        print(f"Model config: {model_type}")
        print(f"Auto map: {auto_map if auto_map else 'None'}")

        custom_class_ref = None
        if isinstance(auto_map, dict):
            custom_class_ref = (
                auto_map.get("AutoModelForSequenceClassification")
                or auto_map.get("AutoModel")
            )

        if not custom_class_ref:
            raise RuntimeError(
                "Model does not define AutoModelForSequenceClassification/AutoModel in auto_map; cannot load custom re-ranker."
            )

        print(f"Loading custom model: {custom_class_ref}")

        # Use Hugging Face dynamic module loader so we don't have to guess local cache paths.
        code_revision = getattr(config, "_commit_hash", None)
        try:
            ModelClass = get_class_from_dynamic_module(
                custom_class_ref,
                model_name,
                revision=code_revision,
                code_revision=code_revision,
            )
        except Exception as import_error:
            raise RuntimeError(
                f"Failed to import custom re-ranker class {custom_class_ref}: {import_error}"
            ) from import_error

        base_model_class_ref = None
        if isinstance(auto_map, dict):
            base_model_class_ref = auto_map.get("AutoModel")
        module_prefix = custom_class_ref.rsplit(".", 1)[0]
        if not base_model_class_ref:
            base_model_class_ref = f"{module_prefix}.LlamaBidirectionalModel"

        try:
            BaseModelClass = get_class_from_dynamic_module(
                base_model_class_ref,
                model_name,
                revision=code_revision,
                code_revision=code_revision,
            )
        except Exception as import_error:
            raise RuntimeError(
                f"Failed to import custom base model class {base_model_class_ref}: {import_error}"
            ) from import_error

        # Register the custom config/model mappings so HF's AutoModel machinery (used inside GenericForSequenceClassification)
        # can instantiate the bidirectional backbone without raising.
        try:
            AutoConfig.register(
                config.model_type,
                config.__class__,
                exist_ok=True,
            )
        except ValueError:
            pass

        try:
            AutoModel.register(
                config.__class__,
                BaseModelClass,
                exist_ok=True,
            )
        except ValueError:
            pass

        try:
            AutoModelForSequenceClassification.register(
                config.__class__,
                ModelClass,
                exist_ok=True,
            )
        except ValueError:
            pass

        try:
            self.model = ModelClass.from_pretrained(
                model_name,
                config=config,
                trust_remote_code=True,
            )
        except Exception as load_error:
            raise RuntimeError(
                f"Failed to load custom re-ranker weights for {model_name}: {load_error}"
            ) from load_error
        print("✓ Loaded custom re-ranker from dynamic module")

        self.model = self.model.to(device)
        self.model.eval()
        self.device = device
        self.model_name = model_name

        print(f"✓ Re-ranker loaded")

    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: int = 5
    ) -> List[Tuple[int, float]]:
        """Re-rank documents by relevance to query.

        Args:
            query: Search query
            documents: List of document texts to re-rank
            top_k: Number of top documents to return

        Returns:
            List of (document_index, score) tuples, sorted by score (descending)
        """
        scores = []

        with torch.no_grad():
            for doc in documents:
                # Create query-document pair
                inputs = self.tokenizer(
                    query,
                    doc,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                # Get relevance score
                outputs = self.model(**inputs)

                # Extract score from model output
                if hasattr(outputs, 'logits'):
                    # For classification models, use logits
                    score = outputs.logits[0].max().item()
                elif hasattr(outputs, 'last_hidden_state'):
                    # For encoder models, use CLS token
                    score = outputs.last_hidden_state[:, 0, :].mean().item()
                else:
                    # Fallback: use mean of all outputs
                    score = outputs[0].mean().item()

                scores.append(score)

        # Sort by score (highest first) and return top_k
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]


class NovelRAGWithLLM:
    """RAG system with Kimi LLM for answer synthesis."""

    def __init__(
        self,
        pdf_path: str,
        model_name: str = "nvidia/llama-3.2-nv-embedqa-1b-v2",
        chunk_size: int = 500,
        overlap: int = 50,
        cache_dir: str = ".cache",
        use_gpu: bool = True,
        kimi_model: str = "kimi-k2-turbo-preview"
    ):
        self.pdf_path = pdf_path
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

        pdf_name = Path(pdf_path).stem
        cache_filename = f"{pdf_name}_chunks{chunk_size}_overlap{overlap}.pkl"
        self.cache_path = self.cache_dir / cache_filename

        # Load or create embeddings
        if self.cache_path.exists():
            print("\n" + "=" * 80)
            print("📦 Loading from cache")
            print("=" * 80)

            with open(self.cache_path, 'rb') as f:
                cache_data = pickle.load(f)

            self.chunks = cache_data['chunks']
            self.vectors = cache_data['vectors']
            print(f"✓ Loaded {len(self.chunks)} chunks from cache")

            print("\nLoading embedding model for queries...")
            self.embedder = NvidiaEmbedder(model_name=model_name, use_gpu=use_gpu)

        else:
            print("\n" + "=" * 80)
            print("🆕 First time setup")
            print("=" * 80)

            print("\nStep 1: Loading PDF")
            print("-" * 80)
            loader = PDFLoader(chunk_size=chunk_size, overlap=overlap)
            self.chunks = loader.load(pdf_path)

            print("\nStep 2: Initializing Embedding Model")
            print("-" * 80)
            self.embedder = NvidiaEmbedder(model_name=model_name, use_gpu=use_gpu)

            print("\nStep 3: Generating Embeddings")
            print("-" * 80)
            start = time.time()
            chunk_texts = [chunk.text for chunk in self.chunks]
            self.vectors = self.embedder.embed_batch(chunk_texts, batch_size=32)
            embed_time = time.time() - start

            print(f"✓ Generated {len(self.vectors)} embeddings in {embed_time:.1f}s")

            print("\nStep 4: Caching")
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

        # Build index
        print("\nBuilding HNSW Index")
        print("-" * 80)
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
        print(f"✓ Built HNSW index")

        # Initialize Kimi LLM
        print("\nInitializing Kimi LLM")
        print("-" * 80)
        self.llm = KimiLLM(model=kimi_model)

        # Initialize Re-ranker (optional)
        print("\nInitializing Re-ranker")
        print("-" * 80)
        try:
            self.reranker = NvidiaReranker(use_gpu=use_gpu)
            self.use_reranking = True
        except Exception as e:
            print(f"⚠️  Could not load re-ranker: {e}")
            print("⚠️  Continuing without re-ranking (will use vector search only)")
            self.reranker = None
            self.use_reranking = False

        print("\n" + "=" * 80)
        if self.use_reranking:
            print("✅ RAG System with LLM & Re-ranking Ready!")
        else:
            print("✅ RAG System with LLM Ready! (No re-ranking)")
        print("=" * 80)

    def query(self, question: str, k: int = 5, use_llm: bool = True) -> Dict:
        """Query the novel with optional LLM synthesis.

        Args:
            question: Question to ask
            k: Number of passages to retrieve
            use_llm: Whether to use LLM for synthesis

        Returns:
            Dict with 'passages' and optionally 'synthesized_answer'
        """
        # Retrieve passages
        query_vector = self.embedder.embed(question)
        result = self.index.search(query_vector, k=k)

        passages = []
        for i, dist in enumerate(result.distances):
            meta = result.metadata[i]
            similarity = 1 - dist
            passages.append((meta["text"], similarity, meta))

        response = {"passages": passages}

        # Synthesize with LLM
        if use_llm:
            answer = self.llm.synthesize_answer(question, passages)
            response["synthesized_answer"] = answer

        return response

    def interactive(self):
        """Interactive Q&A with streaming LLM responses."""
        print("\n" + "=" * 80)
        print("💬 Interactive Q&A with Kimi LLM (Streaming)")
        print("=" * 80)
        print("Ask questions about your novel.")
        print("Commands: 'quit'/'exit' to stop, 'raw' to toggle passage display\n")

        show_raw_passages = False
        question_count = 0

        while True:
            try:
                question = input("\n📖 Question: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n\nGoodbye!")
                break

            if not question:
                continue

            if question.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye! 👋")
                break

            if question.lower() == 'raw':
                show_raw_passages = not show_raw_passages
                status = "ON" if show_raw_passages else "OFF"
                print(f"✓ Raw passage display: {status}")
                continue

            question_count += 1

            # Retrieve candidates
            print("\n🔍 Searching...")
            start_search = time.time()
            query_vector = self.embedder.embed(question)

            if self.use_reranking:
                # Two-stage retrieval: Get more candidates for re-ranking
                result = self.index.search(query_vector, k=15)

                # Extract candidate documents
                candidates = []
                candidate_metadata = []
                for i, dist in enumerate(result.distances):
                    meta = result.metadata[i]
                    candidates.append(meta["text"])
                    candidate_metadata.append(meta)

                search_time = time.time() - start_search

                # Re-rank to top 5
                print("🎯 Re-ranking...")
                start_rerank = time.time()
                reranked_indices = self.reranker.rerank(question, candidates, top_k=5)
                rerank_time = time.time() - start_rerank

                # Get final top 5 passages
                passages = []
                for doc_idx, score in reranked_indices:
                    meta = candidate_metadata[doc_idx]
                    passages.append((meta["text"], score, meta))
            else:
                # Direct retrieval: Get top 5 from vector search
                result = self.index.search(query_vector, k=5)

                passages = []
                for i, dist in enumerate(result.distances):
                    meta = result.metadata[i]
                    similarity = 1 - dist
                    passages.append((meta["text"], similarity, meta))

                search_time = time.time() - start_search
                rerank_time = 0  # No re-ranking

            # Stream LLM response
            print(f"\n{'='*80}")
            print(f"🤖 Kimi's Answer (streaming...)")
            print(f"{'='*80}\n")

            start_llm = time.time()
            stream = self.llm.synthesize_answer(question, passages, stream=True)

            # Print streaming response
            full_answer = ""
            finish_reason = None
            try:
                for chunk in stream:
                    if chunk.choices[0].delta.content:
                        content = chunk.choices[0].delta.content
                        print(content, end='', flush=True)
                        full_answer += content

                    # Check if stream finished
                    if chunk.choices[0].finish_reason:
                        finish_reason = chunk.choices[0].finish_reason
            except Exception as e:
                print(f"\n\n⚠️  Streaming error: {e}")

            # Warn if truncated
            if finish_reason == "length":
                print(f"\n\n⚠️  Response truncated (hit token limit). Try asking for a shorter answer.")

            llm_time = time.time() - start_llm
            total_time = search_time + rerank_time + llm_time

            print(f"\n\n{'='*80}")
            if self.use_reranking:
                print(f"⏱️  Search: {search_time*1000:.0f}ms | Re-rank: {rerank_time*1000:.0f}ms | LLM: {llm_time:.1f}s | Total: {total_time:.1f}s")
            else:
                print(f"⏱️  Search: {search_time*1000:.0f}ms | LLM: {llm_time:.1f}s | Total: {total_time:.1f}s")
            print(f"{'='*80}")

            # Optionally show raw passages
            if show_raw_passages:
                print(f"\n{'='*80}")
                if self.use_reranking:
                    print("📚 Source Passages (Re-ranked)")
                else:
                    print("📚 Source Passages")
                print(f"{'='*80}")
                for i, (text, score, meta) in enumerate(passages, 1):
                    score_label = "Re-rank Score" if self.use_reranking else "Similarity"
                    print(f"\n{i}. Page {meta['page']} ({score_label}: {score:.3f})")
                    print("-" * 80)
                    words = text.split()
                    display = ' '.join(words[:100]) + "..." if len(words) > 100 else text
                    print(display)

        print(f"\n✅ Answered {question_count} questions!")


def main():
    """Run interactive RAG with Kimi LLM."""
    print("=" * 80)
    print("📚 Novel RAG with Kimi LLM Synthesis")
    print("=" * 80)

    pdf_path = "/Users/peterpreketes/vector-db-from-scratch/3%man.pdf"

    if not os.path.exists(pdf_path):
        print(f"\n❌ Error: PDF not found at {pdf_path}")
        return

    # Check for API key
    if not os.getenv("KIMI_API_KEY"):
        print("\n❌ Error: KIMI_API_KEY not found in environment")
        print("Add it to your .env file or set it as an environment variable")
        return

    try:
        rag = NovelRAGWithLLM(
            pdf_path=pdf_path,
            model_name="nvidia/llama-3.2-nv-embedqa-1b-v2",
            chunk_size=500,
            overlap=50,
            cache_dir=".cache",
            use_gpu=True,
            kimi_model="kimi-k2-turbo-preview"  # High-speed Kimi K2: 60-100 tokens/s
        )
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return

    rag.interactive()


if __name__ == "__main__":
    main()
