"""Query your novel with command-line questions.

Usage:
    python examples/query_novel.py "What happens in chapter 1?"
    python examples/query_novel.py "Who are the main characters?"
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the NovelRAG class from novel_rag.py
from novel_rag import NovelRAG

def main():
    if len(sys.argv) < 2:
        print("Usage: python query_novel.py \"Your question here\"")
        print("\nExamples:")
        print('  python query_novel.py "What happens at the beginning?"')
        print('  python query_novel.py "Who are the main characters?"')
        print('  python query_novel.py "What is the main conflict?"')
        return

    # Get question from command line
    question = " ".join(sys.argv[1:])

    print("=" * 80)
    print(f"Question: {question}")
    print("=" * 80)

    # Configuration
    pdf_path = "/Users/peterpreketes/vector-db-from-scratch/3%man.pdf"

    # Check file exists
    if not os.path.exists(pdf_path):
        print(f"\nError: PDF not found at {pdf_path}")
        return

    # Initialize RAG system (this will take ~3 minutes on first run)
    print("\nInitializing RAG system...")
    print("(This takes ~3-4 minutes on first run, then it's cached)\n")

    try:
        rag = NovelRAG(
            pdf_path=pdf_path,
            model_name="nvidia/llama-3.2-nv-embedqa-1b-v2",
            chunk_size=500,
            overlap=50
        )
    except Exception as e:
        print(f"\nError: {e}")
        return

    # Query
    print("\nSearching for relevant passages...")
    results = rag.query(question, k=5)

    # Display results
    print("\n" + "=" * 80)
    print(f"Top {len(results)} Relevant Passages")
    print("=" * 80)

    for i, (text, similarity, meta) in enumerate(results, 1):
        print(f"\n{i}. Page {meta['page']}, Chunk {meta['chunk_index']} - Similarity: {similarity:.3f}")
        print("-" * 80)

        # Show relevant excerpt (limit to 200 words for readability)
        words = text.split()
        if len(words) > 200:
            display_text = ' '.join(words[:200]) + "..."
        else:
            display_text = text

        print(display_text)

    print("\n" + "=" * 80)
    print("\nTip: Run again with a different question - the model is now loaded!")


if __name__ == "__main__":
    main()
