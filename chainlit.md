# 🔬 AI Research Assistant

An **agentic RAG-powered** research assistant that breaks your question into focused sub-queries, searches a local vector knowledge base, and synthesizes a structured report with citations.

---

## How it works

1. **Select a backend** when the chat starts
2. **Ask any research question** — the assistant will:
   - Analyze your query and identify sub-topics
   - Search the knowledge base using vector similarity
   - Synthesize a structured report with references
3. **Upload documents** (`.txt .md .pdf .docx .html`) to expand the knowledge base
4. Type **`docs`** to view the current knowledge base

---

## Backends

| Backend | Model | Requirement |
|---------|-------|-------------|
| 🟣 Claude | `claude-sonnet-4-6` | `ANTHROPIC_API_KEY` in `.env` |
| ⚡ OpenAI | `gpt-4o-mini` | `OPENAI_API_KEY` in `.env` |
| 🚀 Groq | `llama-3.3-70b-versatile` | `GROQ_API_KEY` in `.env` |
| 💎 Google Gemini | `gemini-2.0-flash` | `GOOGLE_API_KEY` in `.env` |
| 🖥️ Ollama | configurable | Local Ollama server (`ollama serve`) |
| 🤗 HuggingFace | `distilgpt2` | Local, no API key needed |
| 🔧 Fallback | rule-based | Nothing required |

---

## Web search

The assistant automatically searches the web (DuckDuckGo, or Tavily if `TAVILY_API_KEY` is set) in parallel with the local knowledge base. No configuration needed for DuckDuckGo.

## Example questions to try

- *What are the latest trends in AI safety?*
- *How is deep learning used in NLP?*
- *Explain research methodology in machine learning*
