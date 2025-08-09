# Agentic RAG-Powered AI Research Assistant

Open-source, offline-capable research assistant built for AllyNerds task. No API keys required for local modes. Supports both a native agent pipeline and a LangChain/LangGraph pipeline.

## Specification

### 1) Scope & Objectives
- Understand a research query, break it into sub-queries, retrieve relevant context from a vector DB, and synthesize a structured report with references.
- Run fully offline using local models (Hugging Face or Ollama) and FAISS; or use cloud APIs (Gemini, Groq, OpenAI).
- Simple, transparent agent orchestration with a choice of native or LangGraph implementation.

### 2) Architecture Overview
- Components:
  - `LLMProvider`: Unified interface over Hugging Face (local), Ollama (local), Google Gemini (API), Groq (API), and OpenAI (API).
  - `EmbeddingProvider`: Sentence-Transformers embeddings.
  - `RAGRetriever`: Loads `.txt` docs from `documents/`, chunks, embeds, and builds FAISS index.
  - Agents:
    - `QueryUnderstandingAgent`: infers intent and sub-queries.
    - `PlannerAgent`: proposes steps, focus areas, and output structure.
    - `SearcherAgent`: RAG retrieval + focused summarization for each sub-query.
    - `SynthesizerAgent`: merges results into a final, structured report with references.
  - `AgenticResearchAssistant`: orchestrates the agent chain end-to-end.

- Data flow:
  - User query → QueryUnderstanding → Planner → parallel Searcher calls (RAG) → Synthesizer → Report + Sources.

- Orchestration options:
  - Native agents (default)
  - LangGraph (mirrors native flow using LangChain components; dependencies required)

### 3) RAG Pipeline
- Corpus: `.txt`, `.md`, `.pdf`, `.docx`, `.html` under `documents/` (recursive).
- Chunking: word-based overlapping windows (default 200 words, 40-word overlap).
- Embeddings: `all-MiniLM-L6-v2` (Sentence-Transformers). Falls back to random embeddings if the model cannot load (demo only).
- Similarity: cosine (via normalized inner product) by default; L2 available.
- Retrieval: top-k=3 per sub-query.

### 4) Agents & Prompts (concise)
- `QueryUnderstandingAgent`:
  - Produces JSON: `{ intent, sub_queries[], priority, research_domain }`.
  - Falls back to simple rule-based splitting.
- `PlannerAgent`:
  - Produces JSON: `{ steps[], focus_areas[], output_structure[], complexity }`.
  - Fallback plan provided if parsing fails.
- `SearcherAgent`:
  - Retrieves k=3 chunks via FAISS for each sub-query.
  - Asks LLM to summarize only what’s retrieved, returns `{ content, sources[], relevance_score }`.
- `SynthesizerAgent`:
  - Builds a report using plan + aggregated content.
  - Ensures headings; appends a `References` section from source filenames.

### 5) LLM Backends
- Hugging Face (local, default): auto-selects correct pipeline (`text-generation` vs `text2text-generation` for T5/FLAN family). Default: `distilgpt2`.
- Ollama (local server): configurable model and options, e.g., `mistral:7b-instruct-q5_K_M`.
- Google Gemini (API): `google-generativeai` client; e.g., `gemini-1.5-flash`.
- Groq (API): `groq` client; e.g., `llama3-8b-8192`.
- OpenAI (API): `openai` client; e.g., `gpt-4o-mini`.
- Fallback: rule-based strings when models aren’t available.

### 6) Configuration (Env Vars)
- Hugging Face (local):
  - `HF_MODEL_NAME` (e.g., `google/flan-t5-small`) – optional.
  - Optional private Hub access: `HUGGINGFACE_HUB_TOKEN` or `huggingface-cli login`.
- Ollama (local):
  - `OLLAMA_MODEL` (e.g., `mistral:7b-instruct-q5_K_M`)
  - `OLLAMA_NUM_CTX` (default 8192; e.g., 16000)
  - `OLLAMA_TEMPERATURE` (default 0.3)
- Google Gemini (API):
  - `GOOGLE_API_KEY` (required)
  - `GEMINI_MODEL` (default `gemini-1.5-flash`)
- Groq (API):
  - `GROQ_API_KEY` (required)
  - `GROQ_MODEL` (default `llama3-8b-8192`)
- OpenAI (API):
  - `OPENAI_API_KEY` (required)
  - `OPENAI_MODEL` (default `gpt-4o-mini`)
  - `OPENAI_TEMPERATURE` (default `0.3`)

Note: The app will load a `.env` if `python-dotenv` is installed; otherwise export env vars in your shell.

### 7) CLI Usage
- Install:
```
pip install -r requirements.txt
```
- Run:
```
python agentic_rag_assistant.py
```
- Choose backend:
  - 1) Hugging Face (local)
  - 2) Ollama (local)
  - 3) Fallback (rule-based)
  - 4) Google Gemini (API; requires `GOOGLE_API_KEY`)
  - 5) Groq (API; requires `GROQ_API_KEY`)
  - 6) OpenAI (API; requires `OPENAI_API_KEY`)
- Choose architecture:
  - a) Native agents (default)
  - b) LangGraph (requires `langchain`, `langgraph`, `langchain-community`; these are required when selecting LangGraph mode)
- Menu:
  - Enter custom query
  - Try example queries
  - Add documents to KB (auto re-index)
  - Bulk import folder
  - Download arXiv abstracts
  - View current documents

### 8) Programmatic Usage
```python
import asyncio
from agentic_rag_assistant import AgenticResearchAssistant

async def run_native_ollama():
    assistant = AgenticResearchAssistant(
        llm_provider="ollama", model_name="mistral:7b-instruct-q5_K_M", architecture="native"
    )
    result = await assistant.research("latest trends in AI safety")
    print(result.content)

async def run_langgraph_groq():
    # Requires: export GROQ_API_KEY=... and optional GROQ_MODEL
    assistant = AgenticResearchAssistant(
        llm_provider="groq", model_name=None, architecture="langgraph"
    )
    result = await assistant.research("latest trends in AI safety")
    print(result.content)

asyncio.run(run_native_ollama())
```

### 9) Model Recommendations (Mac, 24 GB RAM)
- Best balance: `mistral:7b-instruct-q5_K_M`
- Also strong: `qwen2.5:7b-instruct-q5_K_M`
- Very fast/small: `phi3.5:mini`
- Avoid heavy models (e.g., 70B or MoE 8x7B) due to memory.

### 10) LangChain/LangGraph Pipeline
- Replaces only orchestration: analyze → plan → search (RAG) → synthesize.
- Uses LC components:
  - Embeddings: `HuggingFaceEmbeddings`
  - Vector store: LC `FAISS` retriever (k=3)
  - LLM: LC `Ollama` or `HuggingFacePipeline`
- Falls back to native mode if LC/LG not installed.

### 11) Performance & Quality Tips
- Increase `OLLAMA_NUM_CTX` (e.g., 16000) for larger retrieved contexts.
- Use q5_K_M quant for 7B–8B models on Mac for best speed/quality tradeoff.
- Ensure `sentence-transformers` is installed to avoid random embedding fallback.

### 12) Limitations
- Basic chunking strategy; no overlap/semantic chunking.
- No web search; limited to local `documents/`.
- Lightweight default HF model yields generic text; prefer a better local or API model.

### 13) File Structure
- `agentic_rag_assistant.py`: all code (agents, RAG, CLI, optional LC/LG)
- `documents/`: `.txt` corpus for RAG
- `requirements.txt`: Python dependencies
- `README.md`: this specification

## Quickstart (condensed)
1) Install: `pip install -r requirements.txt`
2) Choose backend:
   - HF: `export HF_MODEL_NAME=google/flan-t5-small`
   - Ollama: `export OLLAMA_MODEL=mistral:7b-instruct-q5_K_M`
   - Gemini: `export GOOGLE_API_KEY=...` (optional: `GEMINI_MODEL`)
   - Groq: `export GROQ_API_KEY=...` (optional: `GROQ_MODEL`)
3) Run: `python agentic_rag_assistant.py`
4) Pick architecture: native or LangGraph

## Tech
- Python, Transformers, Sentence-Transformers, FAISS, NumPy, Torch
- Optional: LangChain, LangGraph
- Optional: Google Generative AI (Gemini), Groq
