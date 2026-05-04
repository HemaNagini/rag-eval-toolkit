"""
Vector store module.

Wraps FAISS for fast similarity search over chunk embeddings.
Persists the index to disk so we don't re-embed on every run.
"""

import pickle
from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np

from src.ingest import Chunk
from src.embed import embed_texts, embed_query


class VectorStore:
    """
    A FAISS-backed vector store for chunks.
    
    Uses inner-product similarity (cosine, since embeddings are L2-normalized).
    Persists both the FAISS index and the original chunks to disk.
    """
    
    def __init__(self, chunks: List[Chunk] | None = None):
        self.chunks: List[Chunk] = chunks or []
        self.index: faiss.Index | None = None
    
    def build(self) -> None:
        """Embed all chunks and build the FAISS index."""
        if not self.chunks:
            raise ValueError("No chunks to index. Pass chunks to __init__ first.")
        
        print(f"Embedding {len(self.chunks)} chunks...")
        texts = [c.text for c in self.chunks]
        embeddings = embed_texts(texts)
        
        # IndexFlatIP = exact inner-product search.
        # Fine for small corpora (< ~100k chunks). For larger, use IndexIVFFlat.
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings.astype(np.float32))
        
        print(f"Indexed {self.index.ntotal} vectors of dimension {dim}.")
    
    def search(self, query: str, k: int = 3) -> List[Tuple[Chunk, float]]:
        """
        Find the top-k most relevant chunks for a query.
        
        Returns a list of (chunk, similarity_score) tuples, sorted by relevance.
        Similarity scores are in [0, 1] for normalized embeddings.
        """
        if self.index is None:
            raise RuntimeError("Index not built yet. Call .build() first.")
        
        query_vec = embed_query(query).astype(np.float32).reshape(1, -1)
        scores, indices = self.index.search(query_vec, k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:  # FAISS returns -1 if fewer than k results found
                continue
            results.append((self.chunks[idx], float(score)))
        
        return results
    
    def save(self, directory: str) -> None:
        """Persist the index and chunks to a directory."""
        if self.index is None:
            raise RuntimeError("Nothing to save — index not built yet.")
        
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        
        faiss.write_index(self.index, str(path / "faiss.index"))
        with open(path / "chunks.pkl", "wb") as f:
            pickle.dump(self.chunks, f)
        
        print(f"Saved index ({self.index.ntotal} vectors) to {directory}/")
    
    @classmethod
    def load(cls, directory: str) -> "VectorStore":
        """Load a previously saved index from disk."""
        path = Path(directory)
        if not (path / "faiss.index").exists():
            raise FileNotFoundError(f"No index found at {directory}/faiss.index")
        
        store = cls()
        store.index = faiss.read_index(str(path / "faiss.index"))
        with open(path / "chunks.pkl", "rb") as f:
            store.chunks = pickle.load(f)
        
        print(f"Loaded index ({store.index.ntotal} vectors) from {directory}/")
        return store


if __name__ == "__main__":
    # Smoke test: build an index from the corpus and run a few queries
    from src.ingest import load_documents, chunk_documents
    
    docs = load_documents("data/corpus")
    chunks = chunk_documents(docs)
    print(f"Loaded {len(chunks)} chunks from corpus.\n")
    
    store = VectorStore(chunks)
    store.build()
    store.save("data/index")
    
    print("\n" + "=" * 60)
    print("Searching for: 'What is GraphRAG?'")
    print("=" * 60)
    for chunk, score in store.search("What is GraphRAG?", k=3):
        print(f"\nScore: {score:.3f}  |  Source: {chunk.source}")
        print(f"  {chunk.text[:200]}...")
    
    print("\n" + "=" * 60)
    print("Searching for: 'Who created Python?'")
    print("=" * 60)
    for chunk, score in store.search("Who created Python?", k=3):
        print(f"\nScore: {score:.3f}  |  Source: {chunk.source}")
        print(f"  {chunk.text[:200]}...")
    
    print("\n" + "=" * 60)
    print("Searching for: 'How do you reduce hallucinations?'")
    print("=" * 60)
    for chunk, score in store.search("How do you reduce hallucinations?", k=3):
        print(f"\nScore: {score:.3f}  |  Source: {chunk.source}")
        print(f"  {chunk.text[:200]}...")