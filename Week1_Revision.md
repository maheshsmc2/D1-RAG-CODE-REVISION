# Week‑1 Complete Revision — RAG Foundations (Day‑1 → Day‑7)

**Teaching Notebook • Comparison • ASCII Story • Key Code Lines • Interview Takeaways**


## Unified ASCII of All 7 Days

```text
┌────────────────────────────── Week‑1 Unified RAG System ──────────────────────────────┐
|                                                                                       |
|  Ingestion → Chunking → Embeddings → Vector Store → Retriever → LLM → Answer → Eval   |
|                                                                                       |
|  [Docs] ──► [Chunker 800/120] ──► [Embeddings] ──► [FAISS/Milvus] ──► [Retriever k]   |
|                                                   │                    │              |
|                                                   └───── Day‑2/5 ─────┘              |
|                                                                      ▼               |
|                                                         [LLM (temp=0)]               |
|                                                                      ▼               |
|                                                           [Answer/Context]           |
|                                                                      ▼               |
|                                                       [Evaluation (EM / F1)]         |
|                                                                                       |
|  Day‑1: Chunking+Embeddings  |  Day‑2: FAISS  |  Day‑3: QA chain  |  Day‑5: Milvus   |
|  Day‑6: Modular utils/app    |  Day‑7: Guardrails/Gradio           |  Day‑4: Contrast |
└───────────────────────────────────────────────────────────────────────────────────────┘
```

## Comparison Table (Day‑1 → Day‑7)

| Day | Project Focus | Tech / DB | Strength | Weakness | Unique Contribution |
|---|---|---|---|---|---|
| 1 | Chunking + Embeddings | OpenAIEmbeddings | Sets the heart (semantic vectors) | No retrieval/LLM yet | Stable chunk/embedding policy |
| 2 | FAISS Retrieval | FAISS | Fast local ANN search | No QA, only retrieval | Evidence bandwidth (k) introduced |
| 3 | Full RAG v0.5 | FAISS + LLM | End‑to‑end QA | No eval; basic prompt | First working pipeline |
| 4 | Classifier vs RAG | TF‑IDF + PA vs RAG | Boundary clarity | Side track for RAG evolution | Label vs evidence contrast |
| 5 | Milvus + Eval | Milvus + Evaluator | Scale + metrics habit | Heavier infra | Project maturity with accuracy |
| 6 | RAG v1.0 Polished | Modular repo | Prod‑ish structure | Retriever still basic | Separation of concerns |
| 7 | Consolidation | — | Docs, guardrails, demo | No new feature | Portfolio‑ready polish |


### Day‑1 — Chunking & Embeddings (Heart starts)

```python
Why it matters:
• 800/120 balances coherence vs overlap; keeps multiple chunks in context windows.
• Fix an embedding model for week‑long stability (same vector distribution).

Important lines:
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
chunks = splitter.split_documents(docs)

emb = OpenAIEmbeddings(model="text-embedding-3-small")
vecs = emb.embed_documents([c.page_content for c in chunks])
```


### Day‑2 — FAISS Index + Retrieval (Ribs take shape)

```python
Why it matters:
• Introduces searchable ANN index and evidence bandwidth (k).
• Save/load makes prompt and k sweeps reproducible.

Important lines:
from langchain_community.vectorstores import FAISS

db = FAISS.from_documents(chunks, OpenAIEmbeddings())
retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 4})

db.save_local("faiss_index_week1")
db = FAISS.load_local("faiss_index_week1", OpenAIEmbeddings(), allow_dangerous_deserialization=True)
```


### Day‑3 — First Full RAG Pipeline (v0.5)

```python
Why it matters:
• Baseline QA chain with custom prompt & temperature=0 for deterministic eval.

Important lines:
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

prompt = PromptTemplate.from_template(
    "Use the context to answer. Be concise.\nContext:\n{context}\n\nQ: {question}\nA:"
)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": prompt}
)

answer = qa.invoke({"query": "What is RAG and why chunk overlap matters?"})
```


### Day‑4 — Classifier vs RAG (Know the boundary)

```python
Why it matters:
• Shows label prediction ≠ evidence‑grounded explanation; clarifies RAG’s role.

Important lines:
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.pipeline import Pipeline

clf = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1,2), max_df=0.9)),
    ("pa", PassiveAggressiveClassifier(C=0.5, max_iter=1000))
])
clf.fit(train_texts, train_labels)
pred = clf.predict([headline])[0]

rag_answer = qa.invoke({"query": headline})
```


### Day‑5 — Milvus + Evaluation Harness (Scale + Measure)

```python
Why it matters:
• Scales the store and, crucially, adds measurement habit (accuracy/EM).

Important lines:
from langchain_milvus import Milvus
from pymilvus import connections

connections.connect(alias="default", host="127.0.0.1", port="19530")
milvus = Milvus.from_documents(
    chunks,
    OpenAIEmbeddings(),
    collection_name="week1_rag",
    connection_args={"host": "127.0.0.1", "port": "19530"}
)
retriever = milvus.as_retriever(search_kwargs={"k": 5})

from datasets import load_dataset
dataset = load_dataset("json", data_files="eval_set.jsonl")["train"]

def exact_match(y_true, y_pred):
    return int(y_true.strip().lower() == y_pred.strip().lower())

scores = []
for row in dataset:
    pred = qa.invoke({"query": row["question"]})
    pred = pred["result"] if isinstance(pred, dict) else pred
    scores.append(exact_match(row["answer"], pred))
acc = sum(scores) / len(scores)
print(f"Eval Accuracy: {acc:.3f}")
```


### Day‑6 — RAG v1.0 (Modular & Production‑ish)

```python
Why it matters:
• Separation of concerns + single entrypoint; easier tests, swaps, telemetry.

Important lines (utils.py):
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

def make_chunks(docs, size=800, overlap=120):
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    sp = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap)
    return sp.split_documents(docs)

def build_store(chunks, backend="faiss"):
    emb = OpenAIEmbeddings()
    if backend == "faiss":
        return FAISS.from_documents(chunks, emb)
    raise ValueError("Unsupported backend")

def make_qa(retriever, model="gpt-4o-mini", temp=0):
    prompt = PromptTemplate.from_template(
        "Answer using only the context.\nContext:\n{context}\n\nQ:{question}\nA:"
    )
    llm = ChatOpenAI(model=model, temperature=temp)
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever,
                                       chain_type="stuff",
                                       chain_type_kwargs={"prompt": prompt})

Important lines (app.py):
from utils import make_chunks, build_store, make_qa

chunks = make_chunks(docs)
store = build_store(chunks, backend="faiss")
qa = make_qa(store.as_retriever(search_kwargs={"k": 4}))

def ask(q: str) -> str:
    out = qa.invoke({"query": q})
    return out["result"] if isinstance(out, dict) else out
```


### Day‑7 — Guardrails + UI (Portfolio Polish)

```python
Why it matters:
• Demo‑friendly; resilient to transient failures; recruiter‑ready artifact.

Important lines:
def safe_ask(q: str, default="I need more context to answer that."):
    try:
        out = qa.invoke({"query": q})
        return out["result"] if isinstance(out, dict) else out
    except Exception as e:
        print(f"[WARN] QA failed: {e}")
        return default

import gradio as gr
def ui_answer(q):
    return safe_ask(q)

demo = gr.Interface(fn=ui_answer, inputs="text", outputs="text",
                    title="Week‑1 RAG",
                    description="Ask questions based on your corpus.")
# demo.launch()  # Use on local, or Spaces with host/port
```


## Interview Takeaways

- Determinism for eval: temperature=0, fixed chunking, fixed embeddings.
- Repro habit: save/load FAISS; named Milvus collections.
- Baseline first: 'stuff' chain; TF‑IDF+PassiveAggressive for contrast.
- Single entrypoint (ask/safe_ask): logging/caching/guardrails point.
- Always measure something (EM/accuracy) before/after any change.