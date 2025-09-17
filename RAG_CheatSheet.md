# 📘 RAG Day-1 Cheat Sheet (Bits 1–8)

This is a condensed **code-first reference** for RAG pipeline Day-1.  
Use in **Notion** or **GitHub** (Markdown-ready).

---

## BIT-1: `chunk_text` (chunking)
```python
assert max_chars > 0 and 0 <= overlap < max_chars, "Invalid chunk sizes"
cut = s.rfind(sep, start, end) or end            # prefer natural boundary
parts.append(s[start:cut].strip())               # emit chunk
start = max(cut - overlap, start + 1)            # ensure overlap
text = " ".join(text.split())                    # normalize whitespace
```

---

## BIT-2: `build_embeddings_and_index` (embeddings + FAISS)
```python
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
embs = model.encode(chunks, batch_size=64, normalize_embeddings=True)
X = np.asarray(embs, dtype="float32")            # (N, D), unit-norm
index = faiss.IndexFlatIP(X.shape[1])            # IP ≈ cosine (with norm)
index.add(X)                                     # add all vectors
```

---

## BIT-3: `save/load index + metadata` (persistence)
```python
faiss.write_index(index, f"{out_dir}/faiss.index")
np.save(f"{out_dir}/embeddings.npy", X)
with open(f"{out_dir}/meta.jsonl", "w", encoding="utf-8") as f:
    for m in metas: f.write(json.dumps(asdict(m)) + "\n")

index = faiss.read_index(f"{out_dir}/faiss.index")
X = np.load(f"{out_dir}/embeddings.npy")
metas = [json.loads(l) for l in open(f"{out_dir}/meta.jsonl", encoding="utf-8")]
assert index.ntotal == X.shape[0] == len(metas), "Size mismatch"
```

---

## BIT-4: `search_topk` (retrieve + dedup)
```python
qv = encoder.encode([query], normalize_embeddings=True)
D, I = index.search(np.asarray(qv, "float32"), k * 4)  # over-fetch for dedup
for score, idx in zip(D[0], I[0]):
    key = metas[idx].get(dedup_by)
    if key in seen: continue                           # doc-level dedup
    results.append({...,"score": float(score), **metas[idx]})
    if len(results) >= k: break
```

---

## BIT-5: `build_prompt` (prompt assembly + token budget)
```python
head = f"{system_prompt}\n\nUser question:\n{query}\n\nUse the context:\n"
head_tokens = int(len(head) * est_tokens_per_char)
ctx_budget = model_ctx_tokens - head_tokens - 320          # reserve tail

context_str, used = assemble_context(chunks, max_chars_per_chunk=1100)
max_chars_ctx = int(ctx_budget / est_tokens_per_char)
if len(context_str) > max_chars_ctx:                       # greedy trim by blocks
    blocks = context_str.split("\n\n---\n\n")
    ...

user = f"{head}{context_str}\n\nRules:\n- Cite sources like [1].\n- If uncertain, say so."
```

---

## BIT-6: `generate_answer` (LLM call + refusal)
```python
response = client.ChatCompletion.create(
    model=model,
    messages=[{"role":"system","content":prompt["system"]},
              {"role":"user","content":prompt["user"]}],
    temperature=0.0, max_tokens=512,
)
text = response.choices[0].message["content"].strip()
status = "refusal" if "don't know" in text.lower() else "answered"
return {"query": query, "answer": text, "used_chunks": prompt.get("used_chunks", []), "status": status}
```

---

## BIT-7: `eval_harness` (retrieval + answer metrics)
```python
# retrieval
D, I = index.search(np.asarray(qvec, "float32"), k)
hit = any(i in rel for i in I[0])                          # hit@k
rr = next((1.0/r for r,i in enumerate(I[0],1) if i in rel), 0.0)  # MRR piece

# answers (lexical baseline)
y_pred.append(int(gold.lower() in pred.lower()))           # precision/recall/F1
```

---

## BIT-8: `app.py` (glue: build + query loop)
```python
# build
chunks = chunk_text(open(path, encoding="utf-8").read())
metas = [ChunkMeta(doc_id="doc1", chunk_id=i, start=0, end=0, source=path).__dict__ for i,_ in enumerate(chunks)]
index, X = build_embeddings_and_index(chunks)
save_index_and_meta(index, X, metas, out_dir)

# query
index, X, metas = load_index_and_meta(out_dir)
hits = search_topk(q, encoder, index, metas, k=5)
prompt = build_prompt(q, hits)
answer = generate_answer(q, prompt)
print("Answer:", answer["answer"])
```
