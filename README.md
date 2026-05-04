# rag-eval-toolkit

An open-source evaluation framework for Retrieval-Augmented Generation (RAG) systems — featuring LLM-as-judge groundedness scoring, retrieval precision metrics, and CI-integrated regression testing.

> In active development. See [Status](#status) below for current progress.

## Why this exists

Most RAG demos look great in a notebook but break silently in production. A model that hallucinates 5% of the time is a different system than one that hallucinates 30% of the time — but you can't tell which one you have without measuring.

This toolkit gives you the building blocks to know *when* and *how* your RAG system is degrading, before users tell you.

## What it does

- **Loads and chunks documents** with configurable chunk size and overlap
- **Embeds chunks** into a vector store for semantic retrieval
- **Retrieves and generates answers** using an LLM (default: GPT-4o-mini)
- **Scores answer quality** along multiple dimensions:
  - **Groundedness** — is the answer supported by the retrieved context?
  - **Relevance** — does the answer actually address the question?
  - **Retrieval precision** — did the right chunks make it to the LLM?
- **Regression-tests** changes to prompts, models, or pipelines against a baseline
- **CI-integrated** so quality drops fail the build, not production

## Stack

- **Python 3.11**
- **LangChain** for orchestration
- **FAISS** for vector search
- **sentence-transformers** for local embeddings
- **OpenAI API** (GPT-4o-mini) for generation and LLM-as-judge evaluation
- **pytest** for regression testing

## Status

| Component | Status |
|---|---|
| Document ingestion + chunking | ✅ Done |
| Vector store + embedding | 🚧 In progress |
| Retrieval + generation pipeline | ⬜ Planned |
| LLM-as-judge evaluators | ⬜ Planned |
| Regression test harness | ⬜ Planned |
| CI workflow (GitHub Actions) | ⬜ Planned |

## Quick start

> Note: this section will be updated as more components are completed.

```bash
# Clone the repo
git clone https://github.com/HemaNagini/rag-eval-toolkit.git
cd rag-eval-toolkit

# Set up the virtual environment
python -m venv venv
source venv/Scripts/activate   # Windows (Git Bash)
# source venv/bin/activate     # macOS / Linux

# Install dependencies
pip install -r requirements.txt

# Add your OpenAI key
echo "OPENAI_API_KEY=sk-your-key-here" > .env

# Verify the document loader
python -m src.ingest
```

## Project structure
rag-eval-toolkit/
├── data/
│   └── corpus/               # source documents
├── src/
│   └── ingest.py             # document loading and chunking
├── requirements.txt
├── .env.example              # template for OpenAI API key
└── README.md
## Author

**Hema M** — Senior GenAI / AI Engineer
[LinkedIn](https://www.linkedin.com/in/hema-mn12/)

## License

MIT — see [LICENSE](./LICENSE) for details.