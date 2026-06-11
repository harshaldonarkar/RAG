"""
Chainlit UI for the Agentic RAG Research Assistant.
Run with: chainlit run app.py -w
"""

import asyncio
import os
import shutil
import time
from pathlib import Path

import chainlit as cl
from dotenv import load_dotenv

load_dotenv()

from agentic_rag_assistant import AgenticResearchAssistant, ConversationMemory

# ---------------------------------------------------------------------------
# Backend registry
# ---------------------------------------------------------------------------
BACKENDS: dict[str, tuple[str, str | None]] = {
    "claude":      ("Claude (Anthropic)",  os.getenv("CLAUDE_MODEL",  "claude-sonnet-4-6")),
    "openai":      ("OpenAI",              os.getenv("OPENAI_MODEL",  "gpt-4o-mini")),
    "groq":        ("Groq",                os.getenv("GROQ_MODEL",    "llama-3.3-70b-versatile")),
    "gemini":      ("Google Gemini",       os.getenv("GEMINI_MODEL",  "gemini-2.0-flash")),
    "ollama":      ("Ollama (local)",      os.getenv("OLLAMA_MODEL",  "mistral:7b-instruct-q5_K_M")),
    "huggingface": ("HuggingFace (local)", os.getenv("HF_MODEL_NAME", "distilgpt2")),
    "fallback":    ("Fallback (no API)",   None),
}

BACKEND_ENV_REQS: dict[str, str] = {
    "claude": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "groq":   "GROQ_API_KEY",
    "gemini": "GOOGLE_API_KEY",
}

SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf", ".docx", ".html", ".htm"}


# ---------------------------------------------------------------------------
# Chat start — pick backend
# ---------------------------------------------------------------------------
@cl.on_chat_start
async def on_chat_start() -> None:
    await cl.Message(
        content=(
            "# 🔬 AI Research Assistant\n"
            "Agentic RAG · Multi-agent pipeline · Vector search\n\n"
            "---\n"
            "**Select your LLM backend to begin:**"
        ),
    ).send()

    actions = [
        cl.Action(name="claude",      payload={"value": "claude"},      label="🟣 Claude Sonnet (Anthropic)"),
        cl.Action(name="openai",      payload={"value": "openai"},      label="⚡ OpenAI GPT-4o-mini"),
        cl.Action(name="groq",        payload={"value": "groq"},        label="🚀 Groq Llama 3.3 70B"),
        cl.Action(name="gemini",      payload={"value": "gemini"},      label="💎 Google Gemini 2.0 Flash"),
        cl.Action(name="ollama",      payload={"value": "ollama"},      label="🖥️  Ollama (local)"),
        cl.Action(name="huggingface", payload={"value": "huggingface"}, label="🤗 HuggingFace (local)"),
        cl.Action(name="fallback",    payload={"value": "fallback"},    label="🔧 Fallback (no API key needed)"),
    ]

    res = await cl.AskActionMessage(
        content="Choose a backend:",
        actions=actions,
        timeout=120,
    ).send()

    provider = (res.get("payload", {}).get("value") or res.get("name", "fallback")) if res else "fallback"
    label, model = BACKENDS[provider]

    # Warn about missing API keys
    env_req = BACKEND_ENV_REQS.get(provider)
    if env_req and not os.getenv(env_req):
        await cl.Message(
            content=(
                f"⚠️ **{env_req}** not found in environment.\n"
                f"Add it to your `.env` file or export it, then restart.\n"
                f"Falling back to rule-based mode for now."
            )
        ).send()
        provider, model = "fallback", None
        label = "Fallback (no API)"

    init_msg = await cl.Message(content=f"⚙️ Initializing **{label}**…").send()

    try:
        loop = asyncio.get_event_loop()
        assistant: AgenticResearchAssistant = await loop.run_in_executor(
            None,
            lambda: AgenticResearchAssistant(
                llm_provider=provider,
                model_name=model,
                architecture="native",
            ),
        )
        cl.user_session.set("assistant", assistant)
        cl.user_session.set("backend_label", label)
        cl.user_session.set("memory", ConversationMemory())

        await cl.Message(
            content=(
                f"✅ **{label}** ready!\n\n"
                "**What you can do:**\n"
                "- 💬 Type any research question\n"
                "- 💬 Ask follow-up questions — I remember the conversation\n"
                "- 📎 Upload documents (`.txt .md .pdf .docx .html`) to expand the knowledge base\n"
                "- 🔍 Type `docs` to list the current knowledge base\n"
                "- 🗑️ Type `clear` to reset conversation memory\n"
                "- 🔄 Type `switch` to change backend"
            )
        ).send()

    except Exception as exc:
        await cl.Message(
            content=f"❌ Could not initialize **{label}**: `{exc}`\nUsing fallback mode."
        ).send()
        loop = asyncio.get_event_loop()
        assistant = await loop.run_in_executor(
            None,
            lambda: AgenticResearchAssistant(llm_provider="fallback", architecture="native"),
        )
        cl.user_session.set("assistant", assistant)
        cl.user_session.set("backend_label", "Fallback")

    # Remove the "initializing" message (replace with empty to clean up)
    init_msg.content = ""
    await init_msg.update()


# ---------------------------------------------------------------------------
# Message handler
# ---------------------------------------------------------------------------
@cl.on_message
async def on_message(message: cl.Message) -> None:
    assistant: AgenticResearchAssistant | None = cl.user_session.get("assistant")
    if not assistant:
        await cl.Message(content="⚠️ Session expired — please refresh the page.").send()
        return

    # Handle file attachments
    if message.elements:
        await handle_uploads(message, assistant)
        if not message.content.strip():
            return

    query = message.content.strip()
    if not query:
        return

    # Special commands
    if query.lower() in ("docs", "list docs", "show docs", "kb"):
        await show_docs(assistant)
        return

    if query.lower() in ("switch", "change backend", "backend"):
        await cl.Message(
            content="Please refresh the page to select a different backend."
        ).send()
        return

    if query.lower() in ("clear", "clear memory", "reset", "forget"):
        cl.user_session.set("memory", ConversationMemory())
        await cl.Message(content="🗑️ Conversation memory cleared.").send()
        return

    memory: ConversationMemory = cl.user_session.get("memory") or ConversationMemory()

    # Run research
    start = time.time()
    try:
        result, rewritten_query = await run_research(assistant, query, memory)
    except Exception as exc:
        await cl.Message(content=f"❌ Research failed: `{exc}`").send()
        return

    elapsed = time.time() - start

    # Store this turn in memory (use a short summary: first 400 chars of the report)
    summary = result.content[:400].replace("\n", " ")
    memory.add(rewritten_query, summary)
    cl.user_session.set("memory", memory)

    await send_result(result, elapsed)


# ---------------------------------------------------------------------------
# Research pipeline — with chainlit step display
# ---------------------------------------------------------------------------
async def run_research(
    assistant: AgenticResearchAssistant,
    query: str,
    memory: ConversationMemory,
) -> tuple:
    """Run the 4-agent pipeline while emitting Chainlit step cards.

    Returns (ResearchResult, rewritten_query).
    """
    # Step 1 — Query understanding (+ optional rewrite for follow-ups)
    async with cl.Step(name="Query Understanding", type="tool", show_input=True) as step:
        step.input = query

        # Rewrite follow-up queries into standalone queries
        rewritten = query
        if not memory.is_empty():
            rewritten = await assistant.query_agent.rewrite_query(query, memory)

        research_query = await assistant.query_agent.process(rewritten)

        rewrite_note = (
            f"\n\n**Rewritten query:** _{rewritten}_"
            if rewritten != query else ""
        )
        step.output = (
            f"**Intent:** {research_query.intent}{rewrite_note}\n\n"
            f"**Sub-queries:**\n"
            + "\n".join(f"- {sq}" for sq in research_query.sub_queries)
            + f"\n\n**Priority:** {research_query.priority}"
        )

    # Step 2 — Planning
    async with cl.Step(name="Research Planning", type="tool") as step:
        plan = await assistant.planner_agent.process(research_query)
        steps_md = "\n".join(f"- {s}" for s in plan.get("steps", []))
        areas_md = "\n".join(f"- {a}" for a in plan.get("focus_areas", []))
        step.output = (
            f"**Steps:**\n{steps_md}\n\n"
            f"**Focus areas:**\n{areas_md}\n\n"
            f"**Complexity:** {plan.get('complexity', 'medium')}"
        )

    # Step 3 — Knowledge base + web search (parallel)
    async with cl.Step(name="Knowledge Base + Web Search", type="retrieval") as step:
        tasks = [assistant.searcher_agent.process(sq) for sq in research_query.sub_queries]
        search_results = list(await asyncio.gather(*tasks))
        all_sources: set[str] = set()
        total_web = sum(r.get("web_results", 0) for r in search_results)
        for r in search_results:
            all_sources.update(r.get("sources", []))

        rag = assistant.rag_retriever
        ws = assistant.web_searcher
        hybrid = rag.use_hybrid and rag.bm25_index is not None
        reranking = rag.reranker is not None and rag.reranker.model is not None
        hyde = assistant.searcher_agent.use_hyde
        web_on = ws is not None and ws.is_available
        web_engine = "Tavily" if (ws and ws._tavily) else "DuckDuckGo"

        parts = []
        if hyde:
            parts.append("🔮 HyDE")
        parts.append("🔀 hybrid BM25 + dense" if hybrid else "🔍 dense")
        if reranking:
            parts.append("🎯 reranked")
        if web_on:
            parts.append(f"🌐 {web_engine}")
        mode_tag = " → ".join(parts)

        n = len(research_query.sub_queries)
        web_note = f" + **{total_web}** web result(s)" if total_web else ""
        step.output = (
            f"**Mode:** {mode_tag}\n\n"
            f"Searched **{n}** sub-quer{'y' if n == 1 else 'ies'}{web_note}, "
            f"retrieved from **{len(all_sources)}** source(s):\n"
            + "\n".join(f"- `{s}`" for s in sorted(all_sources))
        )

    # Step 4 — Synthesis (with prior conversation context if available)
    async with cl.Step(name="Synthesizing Report", type="tool") as step:
        research_data = {
            "query": rewritten,
            "search_results": search_results,
            "plan": plan,
            "research_query": research_query,
            "prior_context": memory.format_for_synthesis() if not memory.is_empty() else "",
        }
        result = await assistant.synthesizer_agent.process(research_data)
        memory_note = f" · 🧠 {len(memory.turns)} prior turn(s)" if not memory.is_empty() else ""
        step.output = f"Report ready · {len(result.content):,} chars · {len(result.sources)} source(s){memory_note}"

    return result, rewritten


# ---------------------------------------------------------------------------
# Result display
# ---------------------------------------------------------------------------
async def send_result(result, elapsed: float) -> None:
    conf = max(0.0, min(result.confidence, 1.0))
    filled = round(conf * 10)
    bar = "█" * filled + "░" * (10 - filled)

    elements: list[cl.Text] = []
    if result.sources:
        src_md = "\n".join(f"- `{s}`" for s in sorted(result.sources))
        elements.append(
            cl.Text(name="📚 Sources", content=src_md, display="side")
        )

    footer = (
        f"\n\n---\n"
        f"📊 Confidence `{bar}` {conf:.0%}  ·  "
        f"⏱️ {elapsed:.1f}s  ·  "
        f"🔗 {' → '.join(result.agent_chain)}"
    )

    await cl.Message(
        content=result.content + footer,
        elements=elements,
    ).send()


# ---------------------------------------------------------------------------
# File upload handler
# ---------------------------------------------------------------------------
async def handle_uploads(message: cl.Message, assistant: AgenticResearchAssistant) -> None:
    added, skipped = [], []

    for el in message.elements:
        if not getattr(el, "path", None):
            continue
        ext = Path(el.name).suffix.lower()
        if ext in SUPPORTED_EXTENSIONS:
            dst = os.path.join(assistant.rag_retriever.documents_path, el.name)
            shutil.copy2(el.path, dst)
            added.append(el.name)
        else:
            skipped.append(el.name)

    if added:
        loop = asyncio.get_event_loop()
        async with cl.Step(name="Indexing Documents", type="tool") as step:
            await loop.run_in_executor(None, assistant.rag_retriever._load_and_index_documents)
            chunking_mode = "✂️ semantic" if assistant.rag_retriever.semantic_chunker else "✂️ word-based"
            step.output = f"Indexed {len(added)} file(s) · {chunking_mode} chunking"

        files_md = "\n".join(f"- `{f}`" for f in added)
        await cl.Message(
            content=f"✅ **{len(added)} document(s) added** to the knowledge base:\n{files_md}"
        ).send()

    if skipped:
        await cl.Message(
            content=(
                f"⚠️ Skipped {len(skipped)} unsupported file(s): {', '.join(skipped)}\n"
                f"Supported formats: `.txt .md .pdf .docx .html .htm`"
            )
        ).send()


# ---------------------------------------------------------------------------
# Show docs command
# ---------------------------------------------------------------------------
async def show_docs(assistant: AgenticResearchAssistant) -> None:
    docs_path = assistant.rag_retriever.documents_path
    files: list[tuple[str, int]] = []

    for root, _, filenames in os.walk(docs_path):
        for fn in filenames:
            if Path(fn).suffix.lower() in SUPPORTED_EXTENSIONS:
                full = os.path.join(root, fn)
                rel = os.path.relpath(full, docs_path)
                files.append((rel, os.path.getsize(full)))

    if not files:
        await cl.Message(content="📭 Knowledge base is empty. Upload some documents to get started!").send()
        return

    total_kb = sum(s for _, s in files) / 1024
    lines = [f"## 📚 Knowledge Base\n{len(files)} documents · {total_kb:.1f} KB total\n"]
    for rel, size in sorted(files):
        lines.append(f"- `{rel}` ({size:,} B)")

    await cl.Message(content="\n".join(lines)).send()
