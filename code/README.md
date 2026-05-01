# Support Triage Agent (`code/`)

A terminal-based, RAG-grounded triage agent for the HackerRank Orchestrate hackathon. It reads `support_tickets/support_tickets.csv`, classifies and routes each ticket through a two-stage LangGraph pipeline, and writes `support_tickets/output.csv` with five required fields per row: `status`, `product_area`, `response`, `justification`, `request_type`.

The agent answers strictly from the provided corpus under `data/` (HackerRank, Claude, Visa) and escalates anything high-risk, sensitive, or out of scope. A reviewer quality gate runs after triage to catch safety issues, hallucinations, and inconsistencies.

---

## Architecture

### Stage 1 — Triage

```
START → classify → retrieve → [router]
                                  ├── escalate ──┐
                                  └── reply ─────┤
                                                 ↓
                                              format → END
```

### Stage 2 — Reviewer (Quality Gate)

```
triage output → review → [approved | refined | escalated] → END
```

The graph has five triage nodes, one pure-function conditional edge, and one reviewer node:

| Stage | What it does |
|---|---|
| `classify` | Single LLM call. Resolves company (when `None`) and detects risk flags (fraud, billing dispute, score manipulation, prompt injection, etc.). |
| `retrieve` | Top-K (=8) semantic search over the chunked corpus, scoped by inferred company. |
| `router` (edge) | Pure Python. If any risk flag fires OR retrieval is empty → `escalate`; else → `reply`. |
| `reply` | LLM structured-output call grounded only in retrieved passages. Fails → escalate fallback. |
| `escalate` | LLM structured-output call producing an empathetic, non-committal handoff message. |
| `format` | Defensive Pydantic validation of the final `TicketOutput`. |
| `review` | Quality gate: checks safety (no over-promises, no PII), grounding (no hallucinations), and consistency (status/request_type/response alignment). Can approve, refine, or escalate. |

### Reviewer actions

| Action | Effect |
|---|---|
| `approved` | Triage output passes through unchanged. |
| `refined` | Response/justification rewritten to fix grounding or consistency issues. |
| `escalated` | Safety issue found — status forced to `escalated` with a safe, non-committal message. |

### Design principles (SOLID)

- **Single Responsibility** — one file per node; chunker, indexer, retriever, prompts, LLM clients, and cache are all isolated modules.
- **Open/Closed** — adding a new graph step, a new provider, or a new escalation trigger never modifies existing nodes.
- **Liskov** — all nodes inherit `Node` and are interchangeable in graph wiring.
- **Interface Segregation** — `IRetriever` and `ILLMClient` are minimal, narrow protocols.
- **Dependency Inversion** — the graph builders are the single composition root that inject concrete clients into nodes.

---

## File layout

```
code/
├── README.md                 # this file
├── requirements.txt          # pinned deps
├── .env.example              # copy to .env, set provider API keys
├── main.py                   # entry point (composition root)
├── config.py                 # paths, per-node LLM configs, escalation triggers
│
├── schemas/
│   ├── state.py              # TicketState (LangGraph TypedDict)
│   └── ticket.py             # TicketInput, TicketOutput, ReviewOutput, Chunk (Pydantic)
│
├── agents/
│   ├── triage/               # Five-node triage pipeline
│   │   ├── graph.py          # build_graph(llm_map, retriever) → compiled graph
│   │   └── nodes/
│   │       ├── base.py       # Node ABC
│   │       ├── classify.py
│   │       ├── retrieve.py
│   │       ├── router.py     # pure conditional edge
│   │       ├── reply.py
│   │       ├── escalate.py
│   │       └── format_output.py
│   └── reviewer/             # Quality-gate agent
│       ├── graph.py          # build_reviewer_graph(llm) → compiled graph
│       └── nodes/
│           └── review.py     # safety / grounding / consistency checks
│
├── retrieval/
│   ├── interfaces.py         # IChunker, IRetriever protocols
│   ├── chunker.py            # MarkdownChunker (split at H2)
│   ├── indexer.py            # build_index() over data/*.md
│   └── retriever.py          # ChromaRetriever
│
├── llm/
│   ├── interfaces.py         # ILLMClient protocol
│   ├── factory.py            # create_llm_client(LLMConfig) → ILLMClient
│   ├── prompts.py            # all prompt templates
│   └── clients/
│       ├── mistral_client.py # mistralai wrapper (rate-limit backoff)
│       ├── claude_client.py  # Anthropic Claude wrapper
│       ├── openai_client.py  # OpenAI wrapper
│       └── gemini_client.py  # Google Gemini wrapper
│
├── cache/
│   └── semantic_cache.py     # ChromaDB-backed semantic cache for LLM responses
│
└── utils/
    └── agent_logger.py       # AGENTS.md §5 log writer
```

---

## Setup

Requires Python 3.11+.

```powershell
# from the repo root
python -m venv .venv
.\.venv\Scripts\Activate.ps1     # macOS/Linux: source .venv/bin/activate
pip install -r code/requirements.txt
copy code\.env.example code\.env # then set your API keys in .env
```

The first run will:
1. Walk `data/` (~774 markdown files), chunk at H2 headings, and embed locally with `sentence-transformers/all-MiniLM-L6-v2` (~2 minutes on CPU, no API needed for indexing).
2. Persist the vector index to `code/chroma_db/`.
3. Process every row of `support_tickets/support_tickets.csv` through the two-stage pipeline.
4. Write `support_tickets/output.csv`.

---

## Run

From the repo root (so the `code` package resolves):

```powershell
# normal run (reuses index if present)
python -m code.main

# rebuild the index from scratch
python -m code.main --reindex

# clear the semantic cache before processing
python -m code.main --reset-cache

# debug: process only the first 3 rows
python -m code.main --limit 3
```

CLI flags:

| Flag | Default | Purpose |
|---|---|---|
| `--reindex` | off | Wipe and rebuild the Chroma index |
| `--reset-cache` | off | Clear the semantic LLM response cache before processing |
| `--limit N` | all rows | Process only the first N tickets |
| `--input PATH` | `support_tickets/support_tickets.csv` | Override input CSV |
| `--output PATH` | `support_tickets/output.csv` | Override output CSV |

---

## Multi-provider LLM support

Each node independently selects its LLM provider and model via `config.py`:

```python
# config.py — per-node LLM configuration
CLASSIFY_LLM  = LLMConfig(provider="mistral", model="mistral-small-latest")
REPLY_LLM     = LLMConfig(provider="mistral", model="mistral-small-latest")
ESCALATE_LLM  = LLMConfig(provider="mistral", model="mistral-small-latest")
REVIEWER_LLM  = LLMConfig(provider="mistral", model="mistral-large-latest")
```

Supported providers and their required env vars:

| Provider | Env var | Notes |
|---|---|---|
| `mistral` | `MISTRAL_API_KEY` | Structured output via native JSON schema; rate-limit backoff built in |
| `claude` | `ANTHROPIC_API_KEY` | JSON schema hint + Pydantic retry fallback |
| `openai` | `OPENAI_API_KEY` | Native strict schema for `gpt-4o` family; JSON-mode fallback for older models |
| `gemini` | `GEMINI_API_KEY` | Native `response_schema` for capable models; JSON-mode fallback otherwise |

Only install the SDK for the provider(s) you actually use. All four are listed in `requirements.txt`.

To switch a node to a different provider, edit the corresponding `LLMConfig` in `config.py` — no node code changes required.

---

## Semantic cache

`cache/semantic_cache.py` wraps a dedicated ChromaDB collection (`llm_response_cache`) that stores full triage+reviewer results keyed by ticket embedding. A cached result is returned when cosine similarity to a previous ticket exceeds the threshold (`SEMANTIC_CACHE_THRESHOLD = 0.93`).

At the end of each run the terminal output shows:

```
[cache] 12/50 ticket(s) served from semantic cache
```

Use `--reset-cache` to clear the cache between independent benchmark runs.

---

## Determinism

- `temperature=0` on every LLM call.
- Embedding model and version are pinned in `requirements.txt`.
- Same `CHROMA_DIR` produces identical retrieval across runs.

Two consecutive `python -m code.main` runs over the same input and index should produce byte-identical output CSVs (modulo provider-side non-determinism that occasionally leaks even at temperature 0).

---

## Escalation policy

The agent escalates whenever:
- the classifier raised any flag in `config.ESCALATE_TRIGGERS` (billing disputes, fraud, identity theft, account access by non-owner, security vulnerabilities, legal/regulatory, score manipulation, sitewide outage claims, prompt injection, PII requests),
- retrieval returned zero relevant passages,
- the reply generation failed validation,
- final output validation failed, or
- the reviewer detected a safety issue in the triage output.

The escalation message is empathetic but never promises specific outcomes (refunds, restorations, score changes).

---

## Swapping providers or vector stores

Because nodes depend on `ILLMClient` and `IRetriever`, replacing any LLM provider — or ChromaDB with another vector store — is one new class plus one `LLMConfig` line in `config.py`. No node code changes.

---

## Logging

`utils/agent_logger.py` writes session-start and event entries to `~/hackerrank_orchestrate/log.txt` per AGENTS.md §5. The log is append-only, UTF-8 LF, never committed.
