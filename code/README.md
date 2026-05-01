# Support Triage Agent (`code/`)

A terminal-based, RAG-grounded triage agent for the HackerRank Orchestrate hackathon. It reads `support_tickets/support_tickets.csv`, classifies and routes each ticket through a LangGraph state machine, and writes `support_tickets/output.csv` with five required fields per row: `status`, `product_area`, `response`, `justification`, `request_type`.

The agent answers strictly from the provided corpus under `data/` (HackerRank, Claude, Visa) and escalates anything high-risk, sensitive, or out of scope.

---

## Architecture

```
START → classify → retrieve → [router]
                                  ├── escalate ──┐
                                  └── reply ─────┤
                                                 ↓
                                              format → END
```

The graph has five nodes plus a pure-function conditional edge:

| Stage | What it does |
|---|---|
| `classify` | Single Mistral call. Resolves company (when `None`) and detects risk flags (fraud, billing dispute, score manipulation, prompt injection, etc.). |
| `retrieve` | Top-K (=8) semantic search over the chunked corpus, scoped by inferred company. |
| `router` (edge) | Pure Python. If any risk flag fires OR retrieval is empty → `escalate`; else → `reply`. |
| `reply` | Mistral structured-output call grounded only in retrieved passages. Fails → escalate fallback. |
| `escalate` | Mistral structured-output call producing an empathetic, non-committal handoff message. |
| `format` | Defensive Pydantic validation of the final `TicketOutput`. |

### Design principles (SOLID)

- **Single Responsibility** — one file per node; chunker, indexer, retriever, prompts, and LLM client are all isolated modules.
- **Open/Closed** — adding a new graph step or a new escalation trigger never modifies existing nodes.
- **Liskov** — all nodes inherit `Node` and are interchangeable in graph wiring.
- **Interface Segregation** — `IRetriever` and `ILLMClient` are minimal, narrow protocols.
- **Dependency Inversion** — the graph builder is the single composition root that injects `MistralClient` and `ChromaRetriever` into nodes.

---

## File layout

```
code/
├── README.md                 # this file
├── requirements.txt          # pinned deps
├── .env.example              # copy to .env, set MISTRAL_API_KEY
├── main.py                   # entry point (composition root)
├── config.py                 # paths, model name, escalation triggers
│
├── schemas/
│   ├── state.py              # TicketState (LangGraph TypedDict)
│   └── ticket.py             # TicketInput, TicketOutput, Chunk (Pydantic)
│
├── nodes/                    # one file per node
│   ├── base.py               # Node ABC
│   ├── classify.py
│   ├── retrieve.py
│   ├── router.py             # pure conditional edge
│   ├── reply.py
│   ├── escalate.py
│   └── format_output.py
│
├── retrieval/
│   ├── interfaces.py         # IChunker, IRetriever protocols
│   ├── chunker.py            # MarkdownChunker (split at H2)
│   ├── indexer.py            # build_index() over data/*.md
│   └── retriever.py          # ChromaRetriever
│
├── llm/
│   ├── interfaces.py         # ILLMClient protocol
│   ├── mistral_client.py     # mistralai wrapper, structured output
│   └── prompts.py            # all prompt templates
│
├── graph/
│   └── builder.py            # build_graph(llm, retriever) → compiled graph
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
copy code\.env.example code\.env # then put your MISTRAL_API_KEY in .env
```

The first run will:
1. Walk `data/` (~774 markdown files), chunk at H2 headings, and embed locally with `sentence-transformers/all-MiniLM-L6-v2` (~2 minutes on CPU, no API needed for indexing).
2. Persist the vector index to `code/chroma_db/`.
3. Process every row of `support_tickets/support_tickets.csv` through the graph.
4. Write `support_tickets/output.csv`.

---

## Run

From the repo root (so the `code` package resolves):

```powershell
# normal run (reuses index if present)
python -m code.main

# rebuild the index from scratch
python -m code.main --reindex

# debug: process only the first 3 rows
python -m code.main --limit 3
```

CLI flags:

| Flag | Default | Purpose |
|---|---|---|
| `--reindex` | off | Wipe and rebuild the Chroma index |
| `--limit N` | all rows | Process only the first N tickets |
| `--input PATH` | `support_tickets/support_tickets.csv` | Override input CSV |
| `--output PATH` | `support_tickets/output.csv` | Override output CSV |

---

## Determinism

- `temperature=0` on every Mistral call.
- Embedding model and version are pinned in `requirements.txt`.
- Same `CHROMA_DIR` produces identical retrieval across runs.

Two consecutive `python -m code.main` runs over the same input and index should produce byte-identical output CSVs (modulo Mistral-side non-determinism that occasionally leaks even at temperature 0).

---

## Escalation policy

The agent escalates whenever:
- the classifier raised any flag in `config.ESCALATE_TRIGGERS` (billing disputes, fraud, identity theft, account access by non-owner, security vulnerabilities, legal/regulatory, score manipulation, sitewide outage claims, prompt injection, PII requests),
- retrieval returned zero relevant passages,
- the reply generation failed validation, or
- final output validation failed.

The escalation message is empathetic but never promises specific outcomes (refunds, restorations, score changes).

---

## Swapping providers

Because nodes depend on `ILLMClient` and `IRetriever`, replacing Mistral with another model — or ChromaDB with another vector store — is one new class plus one wiring line in `graph/builder.py`. No node code changes.

---

## Logging

`utils/agent_logger.py` writes session-start and event entries to `~/hackerrank_orchestrate/log.txt` per AGENTS.md §5. The log is append-only, UTF-8 LF, never committed.
