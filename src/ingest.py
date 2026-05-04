"""
Document ingestion module.

Loads .txt files from a corpus folder and splits them into chunks
suitable for embedding and retrieval.
"""

from pathlib import Path
from typing import List, Dict
from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter


@dataclass
class Chunk:
    """A single chunk of text with metadata."""
    text: str
    source: str  # filename it came from
    chunk_id: str  # unique identifier


def load_documents(corpus_dir: str) -> Dict[str, str]:
    """
    Load all .txt files from a directory.
    
    Returns a dict mapping filename -> file contents.
    """
    corpus_path = Path(corpus_dir)
    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus directory not found: {corpus_dir}")
    
    documents = {}
    for txt_file in corpus_path.glob("*.txt"):
        documents[txt_file.name] = txt_file.read_text(encoding="utf-8")
    
    if not documents:
        raise ValueError(f"No .txt files found in {corpus_dir}")
    
    return documents


def chunk_documents(
    documents: Dict[str, str],
    chunk_size: int = 500,
    chunk_overlap: int = 100,
) -> List[Chunk]:
    """
    Split documents into overlapping chunks.
    
    Args:
        documents: dict of filename -> content (from load_documents)
        chunk_size: target characters per chunk
        chunk_overlap: characters of overlap between chunks (preserves context)
    
    Returns:
        List of Chunk objects with source filenames and unique IDs.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    
    all_chunks = []
    for filename, content in documents.items():
        text_chunks = splitter.split_text(content)
        for i, text in enumerate(text_chunks):
            all_chunks.append(Chunk(
                text=text,
                source=filename,
                chunk_id=f"{filename}::chunk_{i}",
            ))
    
    return all_chunks


if __name__ == "__main__":
    # Quick smoke test — run this file directly to verify it works
    docs = load_documents("data/corpus")
    print(f"Loaded {len(docs)} documents:")
    for name in docs:
        print(f"  - {name} ({len(docs[name])} chars)")
    
    chunks = chunk_documents(docs)
    print(f"\nCreated {len(chunks)} chunks total")
    print(f"\nFirst chunk preview:")
    print(f"  Source: {chunks[0].source}")
    print(f"  ID: {chunks[0].chunk_id}")
    print(f"  Text: {chunks[0].text[:200]}...")