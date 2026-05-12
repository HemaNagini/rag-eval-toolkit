"""
Answer generation module.

Sends retrieved chunks to an LLM to produce grounded, source-cited answers.
The system prompt explicitly instructs the model to use only the provided
context, which is the foundation of RAG groundedness.
"""

from typing import List, Tuple
from dataclasses import dataclass
from dotenv import load_dotenv
from openai import OpenAI

from src.ingest import Chunk

# Load OPENAI_API_KEY from .env
load_dotenv()

# Module-level singleton client.
# Reusing the client across calls is more efficient than instantiating each time.
_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI()
    return _client


SYSTEM_PROMPT = """You are a helpful assistant that answers questions strictly using \
the provided context. If the context does not contain enough information to answer, \
say "I don't have enough information to answer that based on the provided context." \
Do not use prior knowledge beyond what is in the context.

After your answer, list the sources you drew from using the format:
Sources: [filename1, filename2]
"""


@dataclass
class Answer:
    """A generated answer with the chunks used to produce it."""
    question: str
    text: str
    chunks_used: List[Chunk]
    model: str


def _format_context(chunks: List[Chunk]) -> str:
    """Format retrieved chunks into a numbered context block for the prompt."""
    parts = []
    for i, chunk in enumerate(chunks, start=1):
        parts.append(f"[{i}] (source: {chunk.source})\n{chunk.text}")
    return "\n\n".join(parts)


def generate_answer(
    question: str,
    chunks: List[Chunk],
    model: str = "gpt-4o-mini",
    temperature: float = 0.0,
) -> Answer:
    """
    Generate a grounded answer to a question using the provided chunks.
    
    Args:
        question: the user's question
        chunks: retrieved chunks from the vector store (typically top-k)
        model: OpenAI model name. gpt-4o-mini is the cost-effective default.
        temperature: 0.0 for deterministic answers (recommended for RAG).
    
    Returns:
        Answer object with the generated text and the chunks used.
    """
    if not chunks:
        return Answer(
            question=question,
            text="I don't have enough information to answer that based on the provided context.",
            chunks_used=[],
            model=model,
        )
    
    context = _format_context(chunks)
    user_prompt = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    
    client = _get_client()
    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )
    
    answer_text = response.choices[0].message.content or ""
    
    return Answer(
        question=question,
        text=answer_text.strip(),
        chunks_used=chunks,
        model=model,
    )


if __name__ == "__main__":
    # Smoke test: synthesize fake chunks and ask a question
    fake_chunks = [
        Chunk(
            text="The capital of France is Paris. It is known for the Eiffel Tower and the Louvre.",
            source="france_facts.txt",
            chunk_id="france_facts.txt::chunk_0",
        ),
        Chunk(
            text="Paris has a population of approximately 2.1 million people in the city proper.",
            source="france_facts.txt",
            chunk_id="france_facts.txt::chunk_1",
        ),
    ]
    
    answer = generate_answer("What is the capital of France?", fake_chunks)
    print(f"Question: {answer.question}\n")
    print(f"Answer: {answer.text}\n")
    print(f"Model: {answer.model}")
    print(f"Chunks used: {len(answer.chunks_used)}")