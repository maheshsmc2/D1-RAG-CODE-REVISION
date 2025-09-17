# 📘 RAG Day-1 — Master Annotated Package (Bits 1–8)

## 🔎 Overview
Purpose: Build a retrieval-augmented QA system with solid engineering:  
- Clean chunking  
- Robust indexing  
- Diversified retrieval  
- Token-aware prompting  
- Guarded generation  
- Measurable evaluation  

```
[text] → chunk → embed → index → retrieve → prompt → LLM → answer
                    └──────── eval_harness (recall@k, MRR, F1) ────────┘
```

---

## BIT-1: `chunk_text` — Chunking with Overlap & Natural Breaks

**Why it matters**  
Converts long text into overlapping, model-friendly chunks. Natural boundaries preserve semantics.

**ASCII Pipeline**
```
[raw text] --normalize--> [clean text]
     │ try \n\n → \n → ". " within window (max_chars)
     ├─ if found: cut at last boundary before end
     └─ else: hard cut (sliding window)
Next start = cut - overlap  (context continuity)
```

**Important Code**
```python
assert max_chars > 0 and 0 <= overlap < max_chars
cut = s.rfind(sep, start, end);  cut = end if cut <= start or cut == -1
parts.append(s[start:cut].strip())
start = max(cut - overlap, start + 1)
text = " ".join(text.split())  # whitespace normalize
```

**Pro Notes**
- Overlap ≈ 10–20% of chunk length  
- Guard against tiny chunks  
- Token-based chunking is more precise  

**Interview Q&A**
- Q: Why overlap?  
  A: Preserves context across cuts.  
- Q: Char vs token chunking?  
  A: Char = simple, token = precise.  

**Mini Quiz**
- What happens if overlap ≥ max_chars?  
- Name one reason chunking can reduce retrieval quality.  

---

## BIT-2: `build_embeddings_and_index` — Embeddings → FAISS

**Why it matters**  
Turns chunks into dense vectors, stores them in FAISS for fast similarity.

**ASCII Pipeline**
```
[chunks] --SentenceTransformer--> [embeddings X (N×D, unit-norm)]
                 │
                 v
            FAISS IndexFlatIP (IP ≈ cosine for unit vectors)
```

**Important Code**
```python
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
X = np.asarray(model.encode(chunks, batch_size=64, normalize_embeddings=True), "float32")
index = faiss.IndexFlatIP(X.shape[1])
index.add(X)
```

**Pro Notes**
- Normalize embeddings before IP  
- Batch_size tuning affects memory  
- IVF/HNSW indices for scale  

**Interview Q&A**
- Q: Why IP with normalized vectors?  
  A: Dot product ≈ cosine.  
- Q: When to use IVF/HNSW?  
  A: Larger datasets.  

**Mini Quiz**
- What goes wrong if vectors aren’t normalized?  
- Which knob to turn first if GPU OOM?  

---

## BIT-3: `save/load index + metadata` — Persistence with Alignment

**Why it matters**  
Keeps index, embedding matrix, and metadata aligned.

**ASCII Pipeline**
```
FAISS index  ─┐
Embeddings X ─┼─ save → {faiss.index, embeddings.npy, meta.jsonl}
Metadata     ─┘
load → (index, X, metas) + assert all sizes match
```

**Important Code**
```python
faiss.write_index(index, f"{out_dir}/faiss.index")
np.save(f"{out_dir}/embeddings.npy", X)
with open(f"{out_dir}/meta.jsonl","w") as f:
    for m in metas: f.write(json.dumps(asdict(m)) + "\n")

index = faiss.read_index(f"{out_dir}/faiss.index")
X = np.load(f"{out_dir}/embeddings.npy")
metas = [json.loads(l) for l in open(f"{out_dir}/meta.jsonl")]
assert index.ntotal == X.shape[0] == len(metas)
```

**Pro Notes**
- Use atomic writes  
- Store versioned directories  
- JSONL is streaming-friendly  

---

## BIT-4: `search_topk` — Query → Candidates with Dedup

**Why it matters**  
Retrieves diverse relevant chunks.

**ASCII Pipeline**
```
[query] → encode → qv
qv → FAISS.search(k×overfetch) → (scores D, ids I)
Dedup by doc_id/source → keep first from each → top-K
```

**Important Code**
```python
qv = encoder.encode([query], normalize_embeddings=True)
D, I = index.search(np.asarray(qv, "float32"), k * 4)
seen, out = set(), []
for score, idx in zip(D[0], I[0]):
    key = metas[idx].get(dedup_by)
    if key in seen: continue
    seen.add(key)
    out.append({**metas[idx], "score": float(score)})
    if len(out) >= k: break
```

**Pro Notes**
- Always over-fetch to allow dedup  
- MMR balances diversity + relevance  

---

## BIT-5: `build_prompt` — Context Assembly

**Why it matters**  
Builds structured prompt with headers and trims to token budget.

**ASCII Pipeline**
```
system + question → estimate head tokens
ctx_budget = ctx_window − head − tail_reserve
assemble_context(chunks) → headers + text
greedy add blocks until fit
```

**Important Code**
```python
head = f"{system}\n\nUser question:\n{query}\n\nUse the context:\n"
ctx_budget = model_ctx_tokens - len(head) - 320
...
user = f"{head}{context_str}\n\nRules:\n- Cite sources like [1].\n- If uncertain, say so."
```

**Pro Notes**
- Reserve space for answer tokens  
- Headers like [1], [2] enforce grounding  

---

## BIT-6: `generate_answer` — LLM Call

**Why it matters**  
Produces safe, cited answers.

**ASCII Pipeline**
```
prompt → LLM
if "don’t know" → refusal
else → answered
```

**Important Code**
```python
resp = client.ChatCompletion.create(...)
text = resp.choices[0].message["content"].strip()
status = "refusal" if "don't know" in text.lower() else "answered"
```

---

## BIT-7: `eval_harness` — Retrieval & Answer Metrics

**Why it matters**  
Measures performance.

**ASCII Pipeline**
```
[queries+gold ids] → retrieve → recall@k, MRR
[gold answers vs pred] → precision, recall, F1
```

**Important Code**
```python
D, I = index.search(np.asarray(qvec, "float32"), k)
hit = any(i in rel for i in I[0])
rr = next((1.0/r for r,i in enumerate(I[0],1) if i in rel), 0.0)
```

---

## BIT-8: `app.py` — Wrapper

**Why it matters**  
Provides CLI/Gradio for end-to-end use.

**ASCII Pipeline**
```
--build file.txt → chunk → embed → index → save
--query query    → load  → search → prompt → LLM → answer
```

**Important Code**
```python
chunks = chunk_text(open(path).read())
index, X = build_embeddings_and_index(chunks)
save_index_and_meta(index, X, metas, out_dir)

hits = search_topk(q, encoder, index, metas, k=5)
prompt = build_prompt(q, hits)
answer = generate_answer(q, prompt)
```

---

## ✅ How to Use

### In **Notion**
1. Copy-paste this Markdown directly into a page → headings, code blocks, and toggles render cleanly.  
2. Or import `.md` → Notion auto-formats it with collapsible headings.  

### In **GitHub**
1. Save as `RAG_Day1_MasterAnnotated.md` in your repo.  
2. GitHub renders it with syntax highlighting and an auto TOC in the sidebar.  
3. You can link to it in your `README.md` for recruiters.  
