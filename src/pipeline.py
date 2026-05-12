"""
End-to-end RAG pipeline.

Orchestrates: load -> chunk -> embed -> retrieve -> generate.
This is the top-level interface most users (and tests) will interact with.
"""

from pathlib import Path
from typing import List

from src.ingest import load_documents, chunk_documents
from src.vectorstore import VectorStore
from src.generate import generate_answer, Answer


class RAGPipeline:
    """
    A full RAG pipeline. Build once, ask many questions.
    
    Usage:
        pipeline = RAGPipeline(corpus_dir="data/corpus", index_dir="data/index")
        pipeline.build()  # or .load() if index already exists
        answer = pipeline.ask("What is GraphRAG?")
        print(answer.text)
    """
    
    def __init__(
        self,
        corpus_dir: str = "data/corpus",
        index_dir: str = "data/index",
        top_k: int = 3,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        model: str = "gpt-4o-mini",
    ):
        self.corpus_dir = corpus_dir
        self.index_dir = index_dir
        self.top_k = top_k
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.model = model
        self.store: VectorStore | None = None
    
    def build(self) -> None:
        """Build the index from scratch and save it to disk."""
        docs = load_documents(self.corpus_dir)
        chunks = chunk_documents(
            docs,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        self.store = VectorStore(chunks)
        self.store.build()
        self.store.save(self.index_dir)
    
    def load(self) -> None:
        """Load a previously built index from disk."""
        self.store = VectorStore.load(self.index_dir)
    
    def build_or_load(self) -> None:
        """Convenience: load if index exists, otherwise build."""
        if Path(self.index_dir, "faiss.index").exists():
            self.load()
        else:
            self.build()
    
    def ask(self, question: str) -> Answer:
        """Ask a question and get a grounded answer."""
        if self.store is None:
            raise RuntimeError("Pipeline not initialized. Call .build() or .load() first.")
        
        retrieved = self.store.search(question, k=self.top_k)
        chunks = [chunk for chunk, _score in retrieved]
        
        return generate_answer(question, chunks, model=self.model)


if __name__ == "__main__":
    # End-to-end smoke test
    pipeline = RAGPipeline()
    pipeline.build_or_load()
    
    test_questions = [
        "What is GraphRAG and how is it different from regular RAG?",
        "Who created Python and when?",
        "How do RAG systems reduce hallucinations?",
        "What is the population of Paris?",  # NOT in our corpus — should refuse
    ]
    
    for q in test_questions:
        print("\n" + "=" * 70)
        print(f"Q: {q}")
        print("=" * 70)
        answer = pipeline.ask(q)
        print(f"\nA: {answer.text}")
        print(f"\n(Used {len(answer.chunks_used)} chunks, model: {answer.model})")