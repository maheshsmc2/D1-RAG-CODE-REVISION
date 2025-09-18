# 📘 Day-3 Teaching Notebook — RAG Code Revision (Role Model)

## 🧭 Pre-Flight — Repo & Run Map

```text
day3_rag_project/
|-- app/
|   \-- app.py              # UI (Gradio/Streamlit) → calls pipeline
|-- pipeline/
|   |-- ingest.py           # 🔴 BIT-1: Chunk docs → embed → store vectors
|   |-- retrieve.py         # 🟡 BIT-2: Retriever (filters, timeouts)
|   |-- rerank.py           # 🟡 BIT-3: Cross-encoder reranker
|   |-- answer.py           # 🔴 BIT-4: Cited answer builder
|   \-- cache.py            # 🟢 BIT-5: Query cache (disk/Redis)
|-- eval/
|   |-- dataset.jsonl       # Eval set
|   |-- harness.py          # Eval harness
|   |-- ab_test.py          # 🟡 BIT-8: Compare rerankers
|   |-- expansion.py        # 🟢 BIT-6: Query expansion
|   \-- reports/            # Reports
|-- configs/
|   \-- rag.yaml            # Config knobs
\-- utils/
    |-- timer.py            # 🔵 BIT-9: Latency budgeting
    \-- safe_call.py        # 🔵 BIT-10: Graceful degradation
```

---

## 🔴 BIT-1 — Deterministic Chunking (Heart)

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(   # 🔑 chunk control
    chunk_size=800,
    chunk_overlap=120,
    separators=["\n\n", ".", "?", "!"]
)
chunks = splitter.split_text(doc)  # → List[str]
```

- 🟢 Dummy-Friendly: Like slicing bread into equal slices with overlaps.
- 🔵 Deep Dive: Input raw doc → output chunks. Deterministic. Params: 800 size, 120 stride.
- ✅ Guarantees: consistent splits
- ⚠️ Breaks if: doc > model max token

```text
[Docs] -> [Chunker — BIT-1] -> [Chunks[]]
```

---

## 🟡 BIT-2 — Retriever with Filters & Timeouts (Ribs)

```python
hits = store.similarity_search(
    query_vec,
    k=10,                          # 🔑 top-k knob
    filter={"source": "docs"},
    timeout=1.5
)
```

- 🟢 Dummy-Friendly: Like a librarian told: “only from history section, back in 2 min.”
- 🔵 Deep Dive: Vector sim (FAISS/Chroma), filters prune irrelevant docs, timeout avoids hangs.
- ✅ Guarantees: fast, scoped results
- ⚠️ Breaks if: timeout too strict -> empty set

```text
[Query Vec] -> [Retriever — BIT-2] -> [Top-k Docs]
```

---

## 🟡 BIT-3 — Cross-Encoder Reranker (Ribs -> Heart)

```python
pairs = [(q, d) for d in hits]
scores = cross_encoder.predict(pairs)   # 🔑 scoring loop
reranked = sorted(zip(hits, scores), key=lambda x: x[1], reverse=True)
```

- 🟢 Dummy-Friendly: Librarian reads blurbs to pick best book.
- 🔵 Deep Dive: Input query+doc pairs, outputs ranked list. Model: MiniLM cross-encoder.
- ✅ Guarantees: better precision
- ⚠️ Breaks if: too slow -> blows latency budget

```text
[Top-k Docs] + [Query] -> [Cross-Encoder — BIT-3] -> [Ranked Docs]
```

---

## 🔴 BIT-4 — Cited Answer Builder (Heart)

```python
context = "\n".join(d.page_content for d in top_docs[:4])  # 🔑 context window
prompt = f"Q: {query}\n\nContext:\n{context}\n\nA:"
answer = llm.generate(prompt)
```

- 🟢 Dummy-Friendly: Writer builds an answer with footnotes.
- 🔵 Deep Dive: Context top 4 docs, citations enforced, fallback 'I don't know'.
- ✅ Guarantees: grounded answers
- ⚠️ Breaks if: context > model token limit

```text
[Ranked Docs] -> [Answer Builder — BIT-4] -> [Cited Answer]
```

---

## 🟢 BIT-5 — Lightweight Cache (Hooks)

```python
key = (query, cfg_hash)   # 🔑 key design
if key in cache:
    return cache[key]
else:
    result = run_pipeline(query)
    cache[key] = result
```

- 🟢 Dummy-Friendly: Sticky notes → if asked yesterday, answer is ready today.
- 🔵 Deep Dive: Cache key = query + config, backends disk or Redis.
- ✅ Guarantees: speed + cost saving
- ⚠️ Breaks if: config changes but cache not invalidated

```text
[Query] -> [Cache — BIT-5] -> [Answer / Pipeline]
```

---

## 🟢 BIT-6 — Query Expansion (Hooks)

```python
if top_docs:   # 🔑 PRF branch
    pseudo_ctx = "\n".join(d.text for d in top_docs[:3])
    prompt = f"Expand '{q}' using:\n{pseudo_ctx}"
else:          # HyDE branch
    prompt = f"Generate hypothetical passage for: {q}"
return llm.generate(prompt).text
```

- 🟢 Dummy-Friendly: Friend rephrases your vague question.
- 🔵 Deep Dive: PRF reformulates using top docs, HyDE hallucinates fake passage.
- ✅ Guarantees: better recall
- ⚠️ Breaks if: expansion adds noise

```text
[Query] -> [Expansion — BIT-6] -> [Expanded Query] -> Retriever
```

---

## 🟡 BIT-7 — Negative Sampling for Eval

```python
negs = store.sample_random(n=2)  # 🔑 random distractors
ex["candidates"] = ex["references"] + [d.id for d in negs]
```

- 🟢 Dummy-Friendly: Teacher adds trick answers in a quiz.
- 🔵 Deep Dive: Augments eval set w/ distractors, tests reranker’s discrimination.
- ✅ Guarantees: fair eval
- ⚠️ Breaks if: negatives too easy/hard

```text
[Eval Set] -> [Add Negatives — BIT-7] -> [Augmented Eval Set]
```

---

## 🟡 BIT-8 — AB Test Rerankers

```python
for name, rer in rerankers.items():   # 🔑 loop compare
    metrics = [score(rer.rerank(q, retriever.query(q)), q) for q in qs]
    results[name] = sum(metrics) / len(metrics)
```

- 🟢 Dummy-Friendly: Taste-test 2 chefs → pick best dish.
- 🔵 Deep Dive: Inputs: queries + reranker models, outputs average metrics per reranker.
- ✅ Guarantees: evidence-based choice
- ⚠️ Breaks if: sample size too small

```text
[Queries] -> [Reranker A — BIT-8] -> [Metrics A]
[Queries] -> [Reranker B — BIT-8] -> [Metrics B]
```

---

## 🔵 BIT-9 — Latency Budgeting

```python
@contextmanager
def timer(label):    # 🔑 p95 tracker
    t0 = time.time()
    yield
    print(label, "took", time.time()-t0, "s")
```

- 🟢 Dummy-Friendly: Stopwatch for each runner in a relay.
- 🔵 Deep Dive: Wrap blocks in timers, SLA p95 < 3.5s.
- ✅ Guarantees: SLA compliance
- ⚠️ Breaks if: only average tracked (hides spikes)

```text
[Pipeline Step] -> [Timer — BIT-9] -> [Latency Report]
```

---

## 🔵 BIT-10 — Error Handling & Graceful Degrade

```python
def safe_call(fn, fallback):   # 🔑 fallback guard
    try:
        return fn()
    except Exception:
        return fallback
```

- 🟢 Dummy-Friendly: Backup generator keeps lights on.
- 🔵 Deep Dive: Wrap risky calls, fallback local model/retriever, prevents crash.
- ✅ Guarantees: no full crash
- ⚠️ Breaks if: fallback too weak

```text
[Function] -> [SafeCall — BIT-10] -> [Result/Fallback]
```

---

## 🔬 Quick Eval Harness (Smoke)

```python
def run_smoke_test(n=3, cfg="configs/rag.yaml"):
    dataset = load_dataset("eval/dataset.jsonl")[:n]
    results = []
    for ex in dataset:
        ans = pipeline.ask(ex["question"], cfg)
        results.append(score(ans, ex["references"]))
    return report(results)
```

```text
[Mini Eval] -> [Pipeline] -> [Scoring EM/F1] -> [Report HTML/JSON]
```

---

## ⚠️ Pitfalls & Debugging Checklist

- Vector dimension mismatch -> FAISS crash
- Timeout too strict -> empty retrievals
- Cache key mismatch -> stale answers
- Context overflow -> silent truncation
- Latency untracked -> SLA violations
- No negatives -> misleading eval

---

## 🚀 End-to-End Pipeline ASCII

```text
Docs -> [BIT-1 Chunker] -> [Embeddings] -> [BIT-2 Retriever] -> [BIT-3 Reranker]
     -> [BIT-4 Answer Builder] -> [BIT-5 Cache]
                 -> [BIT-6 Expansion] -> [Retriever]

Eval: [BIT-7 Negatives] + [BIT-8 AB Test] + [BIT-9 Timer] + [BIT-10 SafeCall]
```

---

## 📑 Quick Sheet (1-Page Revision)

```text
BIT-1: Chunker -> consistent slices
BIT-2: Retriever -> fast, filtered
BIT-3: Reranker -> improves precision
BIT-4: Answer -> citations, fallback
BIT-5: Cache -> saves cost/latency
BIT-6: Expansion -> better recall
BIT-7: Negatives -> fair eval
BIT-8: AB test -> pick best reranker
BIT-9: Timer -> SLA tracking
BIT-10: SafeCall -> resilience
```

**Top 5 Interview Qs**
1. Why chunk docs?
2. FAISS vs Chroma?
3. Why rerankers?
4. Why negatives in eval?
5. What is graceful degrade?
