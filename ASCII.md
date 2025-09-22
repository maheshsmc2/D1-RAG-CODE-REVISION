# Week‑1 Unified RAG — ASCII (Notion)

> Paste or import this page into Notion. The ASCII diagram is inside a code block so it preserves spacing.

## Combined ASCII (Day‑1 → Day‑7)
```text
┌────────────────────────────── Week-1 Unified RAG (Day-1 → Day-7) ───────────────────────────────┐
│                                                                                                 │
│   Ingestion                 Indexing                    Retrieval/Grounding        Generation    │
│                                                                                                 │
│  [Docs]                                                                                               
│     │                                                                                           │
│     ▼                                                                                           │
│  [Chunker 800/120]  ← Day-1 (D1)  ──────────────────────────────────────────────────────────────┤
│     │                                                                                           │
│     ▼                                                                                           │
│  [Embeddings: text-embedding-3-*]  ← D1                                                         │
│     │                                                                                           │
│     ├───────────────►  [FAISS]  ← Day-2 (D2)                                                    │
│     │                 (local ANN; save/load)                                                    │
│     └───────────────►  [Milvus: collection=week1_rag]  ← Day-5 (D5, scale)                      │
│                                 (host/port connect; prod-ish)                                   │
│                                                                                                 │
│                                 ▼                                                               │
│                        [Retriever  k=4..5]  ← D2 baseline; tuned D5                              │
│                                 │                                                               │
│                                 ▼                                                               │
│                      [Prompt (context-first, concise)]  ← Day-3 (D3)                            │
│                                 │                                                               │
│                                 ▼                                                               │
│                      [LLM: ChatOpenAI, temp=0]  ← D3 (deterministic for eval)                   │
│                                 │                                                               │
│                                 ▼                                                               │
│                        [Answer + Cited Context]  ← D3 baseline                                  │
│                                 │                                                               │
│                                 └──► [Evaluation: EM/Accuracy loop]  ← Day-5 (D5)               │
│                                                                                                 │
│  Side Track (Contrast)                                                                           │
│  ───────────────────                                                                               │
│  [TF-IDF] → [PassiveAggressiveClassifier]  ← Day-4 (D4)  == label vs RAG’s evidence-answer       │
│                                                                                                 │
│  Repo & Delivery                                                                                 │
│  ─────────────                                                                                   │
│  [utils.py: make_chunks / build_store / make_qa]  ← Day-6 (D6, modular)                          │
│  [app.py: ask(q)]                                          ← D6                                  │
│  [eval.py: loop/metrics]                                    ← D5/D6                              │
│  [requirements.txt, README]                                 ← D6                                 │
│  [safe_ask try/except]                                      ← Day-7 (D7, guardrails)             │
│  [Gradio UI → demo/Spaces]                                  ← D7                                 │
│                                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────────────────────┘
```
