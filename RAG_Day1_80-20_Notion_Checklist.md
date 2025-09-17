# ✅ RAG Day-1 • 80/20 Code Cheatsheet (Notion Version)

## ▶ Pipeline (Executive Summary)
- [ ] Input question → Embed question → Vector search (top-k) → Retrieve chunks → Prompt LLM → Answer

ASCII Map:
```
[User Q]
   ↓ embed(q)
[q-vec] —— similarity search (k)
   ↓                       ↘
[top-k chunks (text)]       ↘
   ↓                        ↘
[Prompt: system + context] → [LLM] → [Answer]
```

---

## ▶ Vital 20% Code Blocks
- [ ] **Chunking**
```python
def chunk_text(text, chunk_size=500, overlap=50):
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunks.append(text[i:i+chunk_size])
    return chunks
```
- Why it matters: context limits → chunks enable retrieval
- Knobs: `chunk_size`, `overlap`

- [ ] **Embeddings + Vector Store**
```python
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_texts(chunks, embeddings)
```

- [ ] **Retriever**
```python
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
```

- [ ] **RetrievalQA (glue)**
```python
qa = RetrievalQA.from_chain_type(llm=ChatOpenAI(), retriever=retriever)
```

- [ ] **Final Run**
```python
result = qa.run("What is this document about?")
```

---

## ▶ High-Leverage Knobs
- [ ] `k` (2–5 precise, 5–10 exploratory)
- [ ] `chunk_size / overlap`
- [ ] embedding model choice
- [ ] system prompt design

---

## ▶ Debug Checklist (80/20)
- [ ] Retrieval wrong? → Inspect top-k chunks
- [ ] Hallucinations? → Constrain prompt, adjust k/overlap
- [ ] Slow/expensive? → Lower k, smaller chunk_size, lighter LLM
- [ ] Missing answers? → Raise k, different embeddings, better chunking
- [ ] Eval test: 5–10 Q/A pairs, measure hit@k

---

## ▶ Interview Flash Cards
- [ ] Q1: Why chunking?
- [ ] Q2: What is an embedding?
- [ ] Q3: Heart of RAG?
- [ ] Q4: Reduce hallucinations?
- [ ] Q5: Debug retrieval?

---

## ▶ One-Minute Setup (Pseudo)
```python
docs = load_documents(path)
chunks = [c for d in docs for c in chunk_text(d.text)]
vs = FAISS.from_texts(chunks, OpenAIEmbeddings())
qa = RetrievalQA.from_chain_type(llm=ChatOpenAI(), retriever=vs.as_retriever(search_kwargs={"k":3}))
qa.run("Your question here")
```
