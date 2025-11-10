# Vector DB from Scratch — End-to-End Technical Walkthrough

Last updated: 2024-11-09

This document reverse-engineers the entire project from first principles so you can confidently explain (or film) every subsystem—document ingestion, embedding, indexing, retrieval, re-ranking, prompt engineering, and LLM synthesis. Treat it as both a study guide and a long-form script outline for a YouTube deep dive.

---

## 1. Big-Picture Architecture

1. **Document ingestion & caching** – `PDFLoader` chunks a PDF novel into overlapping windows and persists both text and embeddings to `.cache/*.pkl`, so subsequent runs skip preprocessing (`examples/4_complete_rag_system.py:277-421`).
2. **Vector representations** – `NvidiaEmbedder` (Nemotron `llama-3.2-nv-embedqa-1b-v2`) converts chunks into 2048-D float32 embeddings, moved to GPU + FP16 for runtime efficiency (`examples/4_complete_rag_system.py:336-423`).
3. **Index layer** – The handcrafted HNSW implementation in `src/indexes/hnsw.py` is used for approximate nearest-neighbor retrieval, but the repo ships four additional indexes (Linear, KD-Tree, LSH, IVF) for experimentation (`README.md`, `examples/1_basic_index_comparison.py`).
4. **Re-ranking** – A cross-encoder (`nvidia/llama-3.2-nv-rerankqa-1b-v2`) rescoring stage boosts accuracy by judging the top 15 HNSW candidates and returning the best 5 passages (`examples/4_complete_rag_system.py:425-612`).
5. **Generation** – Retrieved context plus a carefully structured prompt feed a Kimi K2 LLM via OpenAI-compatible streaming APIs, optionally augmented with web search tool calls (`examples/4_complete_rag_system.py:165-274` and `prompts/`).
6. **User experience** – `NovelRAGWithLLM.interactive()` handles toggles (web/raw), streaming output, timing telemetry, and graceful shutdown (`examples/4_complete_rag_system.py:700-970`).

---

## 2. Document Ingestion & Storage

- **Cleaner & chunker**: `PDFLoader` removes funky whitespace with regex, strips orphaned page numbers, and slices text by word count so tokens stay under the 4K encoder limit. Overlap (default 50 words) ensures continuity across chunk boundaries (`examples/4_complete_rag_system.py:277-333`).
- **Chunk metadata**: Each `Chunk` carries `page`, `chunk_index`, and `num_words` for traceability during answer citation (`examples/4_complete_rag_system.py:249-275`).
- **Caching strategy**: A deterministic filename per `(pdf, chunk_size, overlap)` stores both text chunks and their embeddings via `pickle`. At runtime we first check `.cache/<pdf>_chunks500_overlap50.pkl`; if present we load vectors directly and only initialize the embedder for query-time use (`examples/4_complete_rag_system.py:642-690`).
- **Storage format**: Plain numpy arrays + dataclasses keep the cache portable, while metadata dictionaries ensure we can reconstruct user-facing text later without re-reading the PDF.

---

## 3. Vector Representation & Distance Functions

- `src/core/vector.py` defines the `Vector` alias (`np.ndarray[np.float32]`) plus helper ops (normalize, dot, magnitude, random sampling). These utilities keep the implementations in `src/indexes` lightweight and numpy-native.
- `src/core/distance.py` centralizes distance metrics (L2, cosine, dot) and exposes both vectorized and scalar routines, so indexes can request their preferred metric without duplicating math.
- `src/core/base.py` introduces `BaseIndex`, `SearchResult`, and `IndexStats`, standardizing APIs (`add`, `search`, `build`, `get_stats`). Every algorithm plugs into this contract, which is why the example scripts can swap indexes with a single import.

---

## 4. Search Algorithms (from Scratch)

| Algorithm | File | Core Idea | Key Implementation Details |
|-----------|------|-----------|----------------------------|
| Linear Scan | `src/indexes/linear_scan.py` | Baseline brute force with 100% recall | Vectorized distance computation, metadata passthrough, excellent for benchmarking correctness |
| KD-Tree | `src/indexes/kdtree.py` | Recursive space partitioning | Uses median splits, alternates axes, prunes branches via bounding boxes; best in ≤20 dims |
| LSH | `src/indexes/lsh.py` | Random projection hashing | Multiple hash tables (bands) increase recall; collisions retrieved before verifying distances |
| IVF | `src/indexes/ivf.py` | Coarse quantization + inverted lists | K-means clustering routes queries to a few centroids, supports multi-probe search |
| HNSW | `src/indexes/hnsw.py` | Hierarchical graph search | Custom level assignment, heuristic neighbor selection, beam search at layer 0; mirrors the 2018 paper |

**Why HNSW for production?**  
It offers log-scale search time, high recall, and is easy to update online. Our version implements:
- Skip-list style level selection (`_select_level`)
- Diverse neighbor heuristic (`_select_neighbors_heuristic`)
- Bidirectional edges to ensure navigability
- Configurable `ef_construction` / `ef_search` to balance speed and recall

Example usage lives in `examples/1_basic_index_comparison.py`, where each algorithm is benchmarked over synthetic data to illustrate trade-offs before moving to real embeddings.

---

## 5. Quantization & Compression

The `src/quantization` package showcases how real vector DBs shrink memory footprints:

1. **Scalar Quantization (`scalar.py`)** – Learns per-dimension min/max, maps float32 → uint8 with linear scaling, yielding 4× compression and faster SIMD-friendly distance calculations.
2. **Product Quantization (`product.py`)** – Splits vectors into subspaces, runs k-means per subspace, and stores centroid codes (8–16× compression). Search uses ADC (Asymmetric Distance Computation) to approximate distances without full reconstruction.
3. **SIMD operations (`simd_ops.py`)** – Houses vectorized dot/L2 kernels to show how low-level optimizations translate to 4–37× throughput improvements.

While the full RAG example keeps embeddings as float32 for clarity, these modules explain how production systems (FAISS, Milvus, etc.) stay within memory budgets.

---

## 6. Embedding Pipeline (Nvidia Nemotron)

- **Model choice**: `nvidia/llama-3.2-nv-embedqa-1b-v2` gives 2048-D embeddings optimized for question answering. It’s lightweight enough (~2.3 GB) to run on Apple MPS or CUDA GPUs.
- **Initialization**: `NvidiaEmbedder` loads the tokenizer and encoder with `torch_dtype=torch.float32`, then moves the model to GPU and halves precision with `.half()` for 2× inference speed and 50% memory savings (`examples/4_complete_rag_system.py:336-383`).
- **Batching**: `embed_batch` tokenizes up to 32 chunks at a time, truncates to 512 tokens, runs the forward pass with `torch.no_grad()`, averages the last hidden state, and L2 normalizes embeddings for cosine-friendly searches (`examples/4_complete_rag_system.py:393-423`).
- **Validation**: A quick “test” inference after loading ensures the model/device combo works and captures the embedding dimension up front.

This mirrors how you would productionize an embedding service: load once, reuse for both ingestion and query-time embeddings, and keep GPU residency to avoid PCIe thrash.

---

## 7. Index Construction & Metadata Binding

- After embeddings are ready, `NovelRAGWithLLM` instantiates an `HNSW` index with `M=16` (graph degree) and `ef_construction=200` for high recall (`examples/4_complete_rag_system.py:716-742`).
- Each chunk’s metadata dictionary is duplicated and enriched (page, chunk index, token count). These metadata objects travel with vectors, so search results can immediately display citations without rejoining external data.
- `index.add()` performs the full HNSW insertion logic: sampling a level, descending from the current entry point, pruning neighbor sets with the paper’s heuristic, and updating `entry_point` when we hit a new max level.

---

## 8. Retrieval Flow (Query-Time)

1. **Embedding** – User questions run through the same `NvidiaEmbedder`, ensuring vector space alignment (`examples/4_complete_rag_system.py:826-838`).
2. **Approximate search** – HNSW retrieves top 15 candidates with `ef_search=50`, balancing latency (~150 ms on MPS) and recall.
3. **Re-ranking** – The top candidates (text + metadata) feed into `NvidiaReranker.rerank()`, which constructs query-document pairs and scores relevance via logits (`examples/4_complete_rag_system.py:580-612`).
4. **Final selection** – We keep the best 5 passages, preserving metadata for downstream display and LLM prompts.

Timing instrumentation prints `Search`, `Re-rank`, `LLM`, and `Total` durations so you can profile each stage live (`examples/4_complete_rag_system.py:912-946`).

---

## 9. Re-ranker Internals & Safetensors Optimization

- **Custom architecture**: NVIDIA ships a `llama_bidirectional_model` (non-causal attention) plus `LlamaBidirectionalForSequenceClassification` head. Loading requires `trust_remote_code=True` and dynamic imports via `transformers.dynamic_module_utils.get_class_from_dynamic_module` (`examples/4_complete_rag_system.py:455-526`).
- **Safetensors workflow**: The repo now prefers a local `model.safetensors` inside the Hugging Face snapshot. We detect it with `huggingface_hub.snapshot_download(..., local_files_only=True)` and pass `use_safetensors=True`, dropping load time from 60 s to ~3 s (`examples/4_complete_rag_system.py:553-585`).
- **Device management**: After loading in FP32 on CPU (less error-prone), we transfer to GPU + FP16 and keep the model in eval mode.
- **Inference**: Each document is scored independently; logits or pooled CLS embeddings become scalar relevance scores. Sorting + slicing returns `(index, score)` tuples for the top `k`.

---

## 10. Generation Layer (Kimi K2 + Prompts)

- **Client**: `KimiLLM` wraps the OpenAI-compatible `OpenAI` SDK but points `base_url` to `https://api.moonshot.ai/v1`. API keys come from `.env` or `KIMI_API_KEY` env var.
- **Prompts**: `prompts/system_prompt.txt` defines the assistant persona (cite pages, ground answers). `prompts/user_prompt_template.txt` injects `{question}` and a formatted list of passages. Editing these files instantly changes LLM behavior—no code changes needed.
- **Request params**: Temperature defaults to 0.3 for grounded answers; `max_tokens`=1000; optional `web_search` toggle adds the `$web_search` builtin tool so the Kimi API can fetch live data when requested.
- **Streaming**: When `stream=True`, the client yields incremental deltas, which we forward to the console for the “word-by-word” UX.

---

## 11. Optional Web Search Tooling

- Typing `web` in the interactive loop toggles a boolean that either attaches or removes the tool definition from API calls (`examples/4_complete_rag_system.py:848-882`).
- Because web search requires multi-turn tool output handling, we warn users that some answers may come back empty and provide a quick toggle-off message.
- This pattern mimics production copilots where retrieval-augmented content is blended with live internet lookups.

---

## 12. Observability & UX

- **Run phases**: The CLI prints banners for cache loading, embedding initialization, index building, LLM init, and re-ranker init, so users always know which stage is active.
- **Stats**: Every question logs milliseconds spent in retrieval/re-ranking and seconds spent in the LLM, which doubles as a lightweight performance monitor.
- **Interactive commands**: `raw` toggles passage display, `web` toggles live search, `quit/exit` ends the session. This makes the script demo-friendly without editing code between takes.

---

## 13. Storage & Dependency Footprint

- Hugging Face cache paths (`~/.cache/huggingface/hub/models--...`) store all model snapshots. We now place `model.safetensors` alongside `pytorch_model.bin` so either path works.
- Project-specific caches (`.cache/*.pkl`) live in the repo root to keep training artifacts under version control if desired (the files are gitignored by default).
- Dependencies stay lightweight: standard scientific Python stack + `torch`, `transformers`, `huggingface-hub`, `pypdf`, `openai`, `python-dotenv`. Optional components (SIMD kernels, advanced quantization) use only numpy/SciPy.

---

## 14. Suggested YouTube Episode Outline

1. **Hook (1 min)** – Show the live RAG demo answering a question with citations, highlight the 8 s cold start after safetensors optimization.
2. **Problem framing** – Why modern search needs vector databases; limitations of keyword search.
3. **From bits to embeddings** – Walk through PDF chunking, metadata, and the Nemotron embedder (visualize a 2048-D vector).
4. **Indexing tour** – Briefly animate linear → KD-Tree → LSH → IVF → HNSW to show the progression from exact to approximate search.
5. **Deep dive: HNSW internals** – Explain layers, entry points, neighbor heuristics; overlay code snippets from `src/indexes/hnsw.py`.
6. **Compression sidebar** – Introduce scalar/product quantization for scale (tie back to FAISS/Qdrant).
7. **Retrieval pipeline** – Illustrate how query embeddings, HNSW search, and Nemotron re-ranking combine before the LLM ever sees text.
8. **Generation & prompts** – Showcase the prompt files, streaming output, and web search toggle.
9. **Performance instrumentation** – Interpret the CLI timing logs; mention GPU/FP16 benefits and safetensors load times.
10. **Takeaways & next steps** – Suggest deploying as an API, swapping in other LLMs, experimenting with web search, or benchmarking other corpora.

Use this document as your speaking notes—each numbered section can become a chapter marker or slide in the video. Highlight the exact file paths on screen so viewers can follow along in the repo.

---

Happy hacking—and happy filming! 🎥
