# Day-5 — RAG Polished Notebook

Complete teaching + revision notebook. Includes all BITs (1–9), end-to-end pipeline, 20 interview Q&A, pitfalls, and cheatsheet. Black-and-white, GoodNotes + Notion-ready.

---

## BIT-1 — Project Map + Run Path

```
day5-rag/
│── app.py              # Streamlit app (UI)
│── config.py           # Central knobs
│── utils.py            # loaders, cleaning, chunking
│── retriever.py        # FAISS build & query
│── pipeline.py         # orchestrates query → retriever → llm
│── llm.py              # LLM response generator
│── eval.py             # evaluation harness
│── requirements.txt
│── README.md
│── data/sample_docs/
│── index/vectorstore/
```

**Flow:** User Query → app.py → pipeline.py → retriever.py → index/vectorstore → llm.py → Final Answer  
**Run Path:** clone repo → install requirements → build index → run app.

---

## BIT-2 — retriever.py

```python
def build_vectorstore(docs, index_path="index/vectorstore"):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(index_path)
    return vectorstore

def query_vectorstore(query, vectorstore, k=3):
    return vectorstore.similarity_search(query, k=k)
```

Handles building, loading, and querying FAISS vector store.

---

## BIT-3 — pipeline.py

```python
def run_pipeline(query, index_path="index/vectorstore"):
    vectorstore = load_vectorstore(index_path)
    docs = query_vectorstore(query, vectorstore, k=3)
    response = get_llm_response(query, docs)
    return response
```

The orchestrator connecting retriever and LLM response.

---

## BIT-4 — llm.py

```python
SYSTEM = "You are a precise assistant. Cite sources as [n]."
def get_llm_response(query, docs, model="gpt-4o-mini"):
    context = _format_context(docs)
    prompt = f"SYSTEM: {SYSTEM}\nQUESTION: {query}\nCONTEXT:\n{context}"
    # call OpenAI or HuggingFace model here
    return response
```

Formats retrieved context, applies SYSTEM rules, and queries OpenAI/HF model.

---

## BIT-5 — app.py

Streamlit UI. Provides input box, sidebar health checks (index + API key), answer display, and self-test expander.

---

## BIT-6 — utils.py

```python
def chunk_docs(docs, chunk_size=800, chunk_overlap=120):
    splitter = RecursiveCharacterTextSplitter(...)
    chunks = []
    for d in docs:
        for i, part in enumerate(splitter.split_text(d.page_content)):
            chunks.append(Document(page_content=part, metadata={**d.metadata, "chunk_id": i}))
    return chunks
```

Handles loading (.txt/.pdf), cleaning text, and chunking docs for embeddings.

---

## BIT-7 — config.py

```python
DATA_DIR = "data/sample_docs"
INDEX_DIR = "index/vectorstore"
CHUNK_SIZE = 800; CHUNK_OVERLAP = 120
TOP_K = 3
LLM_MODEL = "gpt-4o-mini"
TEMPERATURE = 0.2; MAX_TOKENS = 600
```

Centralized constants for tuning RAG system.

---

## BIT-8 — eval.py

Runs sample queries through pipeline and saves results in `eval_results.json`. Helps regression testing.

---

## BIT-9 — requirements.txt + README.md

- **requirements.txt** pins versions (streamlit, langchain, openai, faiss-cpu, torch).  
- **README.md** documents setup, config, evaluation, and project layout.

---

## End-to-End ASCII Pipeline

```
User Query
    │
    ▼
  app.py (UI)
    │
    ▼
 pipeline.py
    │
    ├── retriever.py → FAISS index
    └── llm.py → grounded answer
    ▼
Final Answer → UI
```

---

## Interview Q&A (20)

1. What is RAG? → Retrieval-Augmented Generation with external docs.  
2. Why embeddings vs keyword search? → Captures semantic similarity.  
3. What role does FAISS play? → Vector similarity search.  
4. Why chunk documents? → Fit context window, improve retrieval precision.  
5. Why overlap chunks? → Preserve context continuity.  
6. What if chunks are too small? → Fragments with little meaning.  
7. Why a pipeline wrapper? → Modular, swappable, testable.  
8. How to prevent hallucination? → Ground on context, system rules.  
9. Trade-off of TOP-K? → Recall vs precision vs cost.  
10. How does temperature affect output? → Lower = stable, higher = diverse.  
11. Why store indexes locally? → Save cost, avoid recompute.  
12. How to evaluate RAG quality? → Faithfulness & citation accuracy.  
13. OpenAI vs HF fallback? → API vs offline trade-offs.  
14. Why citation formatting [n]? → Traceability.  
15. Pitfalls in chunking? → Too large = overflow, too small = noise.  
16. When to re-rank docs? → If top-k is noisy.  
17. Can pipeline scale? → Swap FAISS for Milvus, add APIs.  
18. Handling PDFs with bad formatting? → Preprocess/OCR.  
19. Where is system brittle? → Retrieval quality.  
20. What’s next beyond RAG? → Re-ranking, summarization, domain fine-tuning.

---

## Pitfalls

- No index built → run retriever.py first.  
- API key missing → must be set in .env.  
- Chunk size mismatch → retrieval breaks.  
- Long docs → token overflow; fix by tighter chunking/summarization.  
- Eval ignored → regressions creep in.  
- Unpinned deps → ensure reproducibility.  

---

## Cheatsheet (Key Code Lines)

```python
# Build vectorstore
vectorstore = FAISS.from_documents(docs, OpenAIEmbeddings())
vectorstore.save_local("index/vectorstore")

# Load & query
vectorstore = FAISS.load_local("index/vectorstore", OpenAIEmbeddings())
results = vectorstore.similarity_search(query, k=3)

# Pipeline
response = get_llm_response(query, docs)

# LLM prompt
SYSTEM = "Use only CONTEXT. Cite [n]."
temperature=0.2; max_tokens=600
```
