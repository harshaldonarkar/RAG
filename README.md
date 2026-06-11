# Agentic RAG Research Assistant

A multi-agent RAG pipeline built for production-grade document research. Originally developed as part of an NLP Engineer task, then significantly extended with advanced retrieval techniques and multiple LLM backends.

## What it does

Takes a natural language research query, breaks it down intelligently, retrieves relevant content from your documents using hybrid search, reranks results with a cross-encoder, and synthesizes a coherent answer — all through a 4-agent pipeline.

## Architecture

**4-Agent Pipeline (LangGraph orchestration):**

| Agent | Role |
|-------|------|
| `QueryUnderstandingAgent` | Parses intent, rewrites query, generates sub-queries |
| `PlannerAgent` | Decomposes research task into parallel search steps |
| `SearcherAgent` | Executes hybrid retrieval across documents + optional web search |
| `SynthesizerAgent` | Aggregates results into a coherent, sourced answer |

**Retrieval stack:**
- Dense retrieval: FAISS + SentceTransformers embeddings
- Sparse retrieval: BM25Okapi (rank_bm25)
- Hybrid fusion of dense + sparse results
- HyDE (Hypothetical Document Embeddings) for query expansion
- Cross-encoder reranking: MS-MARCO MiniLM (fetches 5× candidates, reranks to top-k)

**LLM backends (6 supported):**
- OpenAI (GPT-4o, GPT-4)
- Anthropic (Claude 3.x)
- Google (Gemini 2.0)
- Groq (Llama 3, Mixtral)
- Ollama (local models)
- HuggingFace Pipelines (local, no API key)

**UI:** Chainlit with step visualization, conversation memory, and query rewriting display.

**Document support:** `.txt`, `.md`, `.pdf`, `.docx`, `.html`

## Setup

```bash
pip install -r requirements.txt
```

Set your preferred backend in `.env`:

```env
LLM_PROVIDER=openai       # openai | anthropic | google | groq | ollama | huggingface
OPENAI_API_KEY=...
# or ANTHROPIC_API_KEY, GOOGLE_API_KEY, GROQ_API_KEY
```

## Running

```bash
# Chainlit UI
chainlit run app.py

# CLI mode
python agentic_rag_assistant.py
```

## Key implementation details

- Chunkingis word-based (200 words, 40 word overlap) — not token-based
- Reranker fetches `k×5` candidates before cross-encoder scoring, ensuring a large enough pool for meaningful reranking
- HyDE uses the hypothetical passage embedding when available, falling back to the original query
- LangGraph used for agent orchestration with both native async and graph-based execution modes
- All agents share a common `LLMProvider` abstraction — swap backends without changing agent code

## Origin

Built as an NLP Engineer task for AllyNerds, then extended with hybrid retrieval, cross-encoder reranking, HyDE, LangGraph orchestration, and multi-backend LLM support.
