# Day-2 RAG Project Revision

**Focus:** Engineering hygiene, observability, deployment, and safe prompting.

---

## BIT 9 — IDs, Persistence, Caching, Determinism
Stable IDs and artifacts make the system reproducible.

```python
def chunk_uid(doc_id, start, end, version="v1"):
    return sha1(f"{version}|{doc_id}|{start}|{end}")[:16]

np.save("embeddings.npy", np.asarray(vecs, dtype="float32"))
```

```
docs -> normalize -> chunk -> embed -> (cache?) -> index -> persist
```

⚠️ **Pitfall:** Switching embedding models mid-corpus corrupts index.

---

## BIT 10 — Observability
Add JSON logs, timers, and metrics.

```python
@contextmanager
def timed(span):
    t0 = time.time()
    yield
    print(f"{span}: {(time.time()-t0)*1000:.2f} ms")
```

```
query -> embed -> search -> rerank -> context
        [log latency] [trace_id]
```

- **Q:** Why add trace_id?  
  **A:** Links logs across pipeline.

---

## BIT 11 — Deployment
Deployment modes: CLI, FastAPI server, batch jobs. Containerize with Docker.

```bash
uvicorn rag_api:app --host 0.0.0.0 --port 8000
```

```
User -> API (/search) -> RAG pipeline -> JSON answer
Batch ingest -> update index -> persist artifacts
```

- **Q:** Why stateless API?  
  **A:** Supports horizontal scaling.

---

## BIT 12 — Prompting & Safety
Prompt templates + refusal modes.

```python
def enforce_citations(text):
    if "[1]" not in text:
        text += "\nLimits: No citations detected."
    return text
```

```
retrieved chunks -> context -> LLM(prompt) -> post-checker -> safe answer
```

- **Q:** Why refusals?  
  **A:** Prevents unsafe or hallucinated answers.

---

## Summary Diagram
```
Day-1 Core:
Raw Text -> Normalize -> Chunk -> Embed -> Index -> Retrieve -> Rerank -> Context -> Answer -> Eval

Day-2 Additions:
IDs -> Cache -> Persistence -> Observability -> Deployment -> Safe Prompting
```

---

## Cheatsheet
- ✅ Carry doc_id + chunk_id end-to-end.  
- ✅ Cache embeddings with model_version key.  
- ✅ Log latency (P50, P95).  
- ✅ Add `/healthz`, `/version` endpoints.  
- ✅ Post-check citations.

---

## Interview Questions
- Q: How do you persist embeddings?  
- Q: Why fix random seeds?  
- Q: How to debug wrong citation IDs?  
- Q: What’s the advantage of Docker for deployment?  
- Q: How do refusal patterns improve trustworthiness?
