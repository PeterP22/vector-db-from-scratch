# Nvidia Nemotron RAG Setup Guide

This guide shows how to use Nvidia's new Nemotron RAG models with your vector database.

## Prerequisites

### 1. Install Dependencies

```bash
pip install torch transformers huggingface-hub
```

For GPU support (recommended):
```bash
# CUDA 11.8
pip install torch --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### 2. Hugging Face Authentication

The Nvidia models require accepting their license and authentication:

1. **Create a Hugging Face account** at https://huggingface.co/join

2. **Accept model licenses** (visit each page and click "Agree"):
   - https://huggingface.co/nvidia/llama-3.2-nv-embedqa-1b-v2
   - https://huggingface.co/nvidia/llama-embed-nemotron-8b
   - https://huggingface.co/nvidia/llama-3.2-nv-rerankqa-1b-v2

3. **Create an access token**:
   - Go to https://huggingface.co/settings/tokens
   - Click "New token"
   - Select "Read" access
   - Copy your token

4. **Login via CLI**:
   ```bash
   pip install huggingface-hub
   huggingface-cli login
   # Paste your token when prompted
   ```

## Available Models

### Embedding Models

| Model | Size | Use Case | Download Size |
|-------|------|----------|---------------|
| `nvidia/llama-3.2-nv-embedqa-1b-v2` | 1B | QA, fast inference | ~2 GB |
| `nvidia/llama-embed-nemotron-8b` | 8B | Best quality | ~16 GB |
| `nvidia/omni-embed-nemotron-3b` | 3B | Multimodal | ~6 GB |

### Reranking Model

| Model | Size | Use Case | Download Size |
|-------|------|----------|---------------|
| `nvidia/llama-3.2-nv-rerankqa-1b-v2` | 1B | Result reranking | ~2 GB |

## Quick Start

### Option 1: Lightweight (1B Model - Recommended for Testing)

```bash
python examples/nvidia_rag_demo.py
```

This uses `nvidia/llama-3.2-nv-embedqa-1b-v2` (1B params, ~2GB download).

**Expected output:**
```
Using device: cuda
Loading nvidia/llama-3.2-nv-embedqa-1b-v2 on cuda...
Model loaded. Embedding dimension: 2048
Generating 10 embeddings...
Building HNSW index...
Running queries...
```

### Option 2: Best Quality (8B Model)

Edit `nvidia_rag_demo.py` line 231:

```python
embedder = NvidiaEmbedder(
    model_name="nvidia/llama-embed-nemotron-8b",  # Change this
    device=device
)
```

**Trade-offs:**
- 1B model: Fast (20-50ms per embedding), good quality, 2GB
- 8B model: Slower (100-200ms per embedding), best quality, 16GB

### Option 3: With Reranking

Uncomment the reranking section in `nvidia_rag_demo.py` (lines 320-340):

```python
try:
    reranker = NvidiaReranker(device=device)

    query_text = queries[0]
    query_vector = embedder.embed(query_text)
    result = index.search(query_vector, k=10)  # Get more candidates

    # Extract document texts
    retrieved_docs = [result.metadata[i]["text"] for i in range(len(result.indices))]

    # Rerank
    reranked = reranker.rerank(query_text, retrieved_docs, top_k=3)

    print("\nReranked results:")
    for i, (doc_idx, score) in enumerate(reranked, 1):
        meta = result.metadata[doc_idx]
        print(f"\n  {i}. [{meta['id']}] - Score: {score:.4f}")
        print(f"     {meta['text'][:80]}...")
except Exception as e:
    print(f"Reranking error: {e}")
```

## Custom Document Integration

### Load Your Own Documents

Replace the `load_sample_documents()` function:

```python
def load_custom_documents(file_path: str) -> List[Document]:
    """Load documents from a text file (one doc per line)."""
    documents = []
    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            if line.strip():
                documents.append(Document(
                    id=f"doc{i}",
                    text=line.strip(),
                    metadata={"source": file_path}
                ))
    return documents

# Usage
documents = load_custom_documents("my_documents.txt")
```

### Load from JSON

```python
import json

def load_from_json(file_path: str) -> List[Document]:
    """Load documents from JSON file."""
    with open(file_path, 'r') as f:
        data = json.load(f)

    documents = []
    for item in data:
        documents.append(Document(
            id=item.get("id", f"doc{len(documents)}"),
            text=item["text"],
            metadata=item.get("metadata", {})
        ))
    return documents
```

## Performance Tips

### 1. GPU vs CPU

**GPU (Recommended):**
```
1B model: ~20-50ms per embedding
8B model: ~100-200ms per embedding
```

**CPU:**
```
1B model: ~200-500ms per embedding
8B model: ~1-3 seconds per embedding
```

### 2. Batch Processing

For large document collections, use batching:

```python
# Generate embeddings in batches
batch_size = 32  # Adjust based on GPU memory
vectors = embedder.embed_batch([doc.text for doc in documents], batch_size=batch_size)
```

### 3. Index Selection

For Nvidia embeddings (dimension ~2048):

| Documents | Recommended Index | Why |
|-----------|------------------|-----|
| < 10K | HNSW | Best recall, fast queries |
| 10K - 100K | HNSW | Still efficient |
| 100K - 1M | IVF + Scalar Quantization | Reduce memory |
| > 1M | IVF + Product Quantization | 16-64x compression |

## Example: Complete RAG Pipeline

```python
from nvidia_rag_demo import NvidiaEmbedder, NvidiaReranker
from src.indexes.hnsw import HNSW

# 1. Initialize models
embedder = NvidiaEmbedder(model_name="nvidia/llama-3.2-nv-embedqa-1b-v2")
reranker = NvidiaReranker()

# 2. Load and embed documents
docs = load_custom_documents("my_docs.txt")
vectors = embedder.embed_batch([doc.text for doc in docs])

# 3. Build index
index = HNSW(dimension=embedder.dimension, M=16, ef_construction=200)
index.add(vectors, [{"id": d.id, "text": d.text} for d in docs])

# 4. Query with reranking
query = "How do I optimize vector search?"
query_vec = embedder.embed(query)

# Get top 20 candidates
result = index.search(query_vec, k=20)
candidates = [result.metadata[i]["text"] for i in range(len(result.indices))]

# Rerank to top 3
reranked = reranker.rerank(query, candidates, top_k=3)

for i, (idx, score) in enumerate(reranked, 1):
    print(f"{i}. {candidates[idx][:100]}... (score: {score:.4f})")
```

## Troubleshooting

### Issue: "Model not found" or "Access denied"

**Solution:** Make sure you've:
1. Accepted the model license on Hugging Face
2. Logged in with `huggingface-cli login`
3. Your token has "Read" permissions

### Issue: "CUDA out of memory"

**Solutions:**
- Reduce batch size: `embedder.embed_batch(texts, batch_size=8)`
- Use smaller model: `nvidia/llama-3.2-nv-embedqa-1b-v2` instead of 8B
- Use CPU: `device="cpu"` (slower)

### Issue: Slow on CPU

**Solutions:**
- Use GPU (highly recommended)
- Use smaller 1B model
- Process documents offline, save index to disk
- Use fewer documents for testing

## Comparison with Other Embedders

| Model | Dimension | Quality | Speed (GPU) | Size |
|-------|-----------|---------|-------------|------|
| Nvidia 1B (QA) | 2048 | Good | 20-50ms | 2 GB |
| Nvidia 8B | 4096 | Excellent | 100-200ms | 16 GB |
| OpenAI ada-002 | 1536 | Excellent | API call | Cloud |
| Sentence-BERT | 384-768 | Good | 5-20ms | 0.5 GB |

**Nvidia Advantages:**
- Open source, run locally
- Optimized for RAG/QA tasks
- Integrated with reranking
- No API costs

## Next Steps

1. **Try the demo**: `python examples/nvidia_rag_demo.py`
2. **Load your documents**: Modify `load_sample_documents()`
3. **Compare models**: Test 1B vs 8B quality
4. **Add reranking**: Uncomment reranker code
5. **Scale up**: Test with larger document collections
6. **Optimize**: Add quantization for memory efficiency

## Resources

- Nvidia Nemotron Collection: https://huggingface.co/collections/nvidia/nemotron-rag
- Vector DB Repo: https://github.com/PeterP22/vector-db-from-scratch
- Hugging Face Docs: https://huggingface.co/docs/transformers
