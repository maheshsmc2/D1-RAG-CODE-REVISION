# 📘 Day-4 — Master Teaching Notebook (Merged RAG + ML, 3C Style)

---

## 🧩 BIT-1 — Dataset / Data Source

```python
# RAG pipeline
with open("data/sample.txt") as f:
    docs = f.readlines()

# ML pipeline
import pandas as pd
df = pd.read_csv("data/sample.csv")
print(df.head())
```

**Explanation**  
- RAG: reads raw text file, basis for chunking.  
- ML: loads CSV with `text`, `label` columns.  

**ASCII Pipeline**
```
RAG: text file → Python list (docs)
ML: CSV → Pandas DataFrame (df)
```

**Interview Notes**
- Why inspect head()? To confirm schema.  
- Text vs label split crucial in supervised tasks.  

**Revision**
- Always confirm data before downstream steps.

---

## 🧩 BIT-2 — Text Cleaning / Preprocessing

```python
# RAG pipeline
def chunk_text(text, size=500):
    return [text[i:i+size] for i in range(0, len(text), size)]

# ML pipeline
def basic_clean(df, text_col):
    df[text_col] = (df[text_col]
                      .fillna("")
                      .str.lower()
                      .str.replace(r"[^a-z0-9\s]", " ", regex=True)
                      .str.replace(r"\s+", " ", regex=True)
                      .str.strip())
    return df
```

**Explanation**  
- RAG: break large text into overlapping chunks for embeddings.  
- ML: normalize (lowercase, strip punctuation, collapse spaces).  

**ASCII Pipeline**
```
Raw Text → [RAG: chunk_text] → Chunks
Raw Text → [ML: clean] → Normalized Text
```

**Interview Notes**
- Why overlap chunks? Preserve context.  
- Why lowercase? Token standardization.  

**Revision**
- Regex cleaning + chunking = core ribs of both pipelines.

---

## 🧩 BIT-3 — Features / Embeddings

```python
# RAG pipeline
embeddings = model.encode(chunks)

# ML pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
vec = TfidfVectorizer(max_features=3000, ngram_range=(1,2))
X = vec.fit_transform(df['text'])
y = df['label']
```

**Explanation**  
- RAG: convert chunks into dense embeddings (vectors).  
- ML: TF-IDF sparse vectors (unigrams+bigrams).  

**ASCII Pipeline**
```
RAG: chunks → dense embeddings
ML: cleaned text → TF-IDF sparse vectors (X)
```

**Interview Notes**
- Dense embeddings = semantic similarity.  
- TF-IDF = interpretable baseline.  

**Revision**
- Both pipelines rely on numerical representations.

---

## 🧩 BIT-4 — Indexing / Splitting

```python
# RAG pipeline
import faiss
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

# ML pipeline
from sklearn.model_selection import train_test_split
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
```

**Explanation**  
- RAG: FAISS builds index for nearest-neighbor search.  
- ML: split into train/test sets (stratified for balance).  

**ASCII Pipeline**
```
RAG: embeddings → FAISS index
ML: X,y → train/test split
```

**Interview Notes**
- Why FAISS? Fast similarity search.  
- Why stratify? Preserve label ratios.  

**Revision**
- Index = retrieval knob. Split = evaluation fairness.

---

## 🧩 BIT-5 — Query / Training

```python
# RAG pipeline
query_vec = model.encode([query])
D, I = index.search(query_vec, k=3)

# ML pipeline
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(max_iter=1000)
clf.fit(Xtr, ytr)
```

**Explanation**  
- RAG: embed query, retrieve top-k chunks.  
- ML: train Logistic Regression classifier.  

**ASCII Pipeline**
```
RAG: query → vector → FAISS search → top-k
ML: X_train, y_train → Logistic Regression model
```

**Interview Notes**
- Why top-k? More evidence improves answers.  
- Why LR baseline? Simple, robust, interpretable.  

**Revision**
- RAG = search, ML = learn decision boundary.

---

## 🧩 BIT-6 — Prompting / Evaluation

```python
# RAG pipeline
context = " ".join([chunks[i] for i in I[0]])
final_prompt = f"Answer with context: {context}\nQuestion: {query}"
response = llm.generate(final_prompt)

# ML pipeline
from sklearn.metrics import accuracy_score, classification_report
preds = clf.predict(Xte)
print("Accuracy:", accuracy_score(yte, preds))
print(classification_report(yte, preds))
```

**Explanation**  
- RAG: assemble context + query → prompt → LLM answer.  
- ML: evaluate predictions against ground truth.  

**ASCII Pipeline**
```
RAG: top-k chunks + query → prompt → LLM → answer
ML: model → predict(X_test) → metrics
```

**Interview Notes**
- Why prompt carefully? To reduce hallucination.  
- Why classification report? Precision/recall > accuracy.  

**Revision**
- Prompt = recipe. Eval = examiner.

---

## 🧩 BIT-7 — Artifacts / Guards

```python
# ML pipeline
def save_artifacts(model, vec, path="artifacts/model.pkl"):
    import joblib, os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump({"model": model, "vec": vec}, path)

def load_artifacts(path="artifacts/model.pkl"):
    import joblib
    bundle = joblib.load(path)
    return bundle["model"], bundle["vec"]

# Defensive guard
def ensure_non_empty(df, col):
    if df[col].isna().all() or (df[col].astype(str).str.len()==0).all():
        raise ValueError(f"Column {col} empty after cleaning")
```

**Explanation**  
- Save model+vectorizer together to prevent mismatch.  
- Guard ensures data not empty post-cleaning.  

**ASCII Pipeline**
```
Model + Vec → joblib.dump → artifacts/model.pkl
```

**Interview Notes**
- Why bundle vec with model? Feature alignment.  
- Why guards? Fail fast on empty data.  

**Revision**
- Persistence + defensive coding ensure reproducibility.

---

## 🧩 BIT-8 — Evaluation Harness (RAG)

```python
def evaluate(query, gold_answer):
    pred = run_pipeline(query)
    return metric(pred, gold_answer)
```

**Explanation**  
- RAG eval: compares generated answer vs reference.  

**ASCII Pipeline**
```
query → pipeline → prediction → compare
```

**Interview Notes**
- Why eval harness? Continuous feedback.  
- Which metrics? Recall@k, F1, faithfulness.  

**Revision**
- Eval = examiner across both pipelines.

---

# 📊 End-to-End ASCII Summary
```
RAG: docs → chunk → embed → FAISS index → query embed → search → prompt → LLM answer → eval
ML: CSV → clean → TF-IDF → split → Logistic Regression → metrics → save artifacts
```

---

# 🎯 Interview Q&A (Combined)
- Why chunk text before embeddings? Context control.  
- Why TF-IDF vs embeddings? Sparse vs dense trade-offs.  
- FAISS vs sklearn split: retrieval vs supervised baseline.  
- How prevent hallucinations? Constrain prompt, cite sources.  
- Why stratify in split? Balanced evaluation.  
- Why bundle vectorizer with model? Avoid mismatch at inference.  
- Alternatives to FAISS? HNSW, Milvus, Weaviate.  
- Scaling beyond TF-IDF? Hashing, embeddings, transformers.  
- Key knobs? Chunk size, k in retrieval, max_features in TF-IDF, C in Logistic Regression.

---

# ⚠️ Pitfalls (Merged)
- RAG: chunk too large → missed context, token overflow.  
- RAG: poor prompt → hallucinations.  
- ML: no stratify → skewed results.  
- ML: vectorizer mismatch → inference crash.  
- Both: no seed → irreproducibility.  

---

# 📘 Day-4 — Final Wrap-Up
**Summary**  
- Two pipelines: Retrieval-Augmented Generation (RAG) + Supervised ML Classification.  
- Common theme: convert raw text → features → model → evaluation.  
- RAG excels at open-domain Q&A, ML pipeline solid for structured classification.  

**Revision Bullets**  
- Inspect data early.  
- Chunking/cleaning = must.  
- Represent text (embeddings or TF-IDF).  
- Index/split = backbone of search/eval.  
- Prompt & Eval define outcomes.  
- Save artifacts with vectorizer.  

**Add-On Insights**  
- RAG future path: rerankers, compression, long-context LLMs.  
- ML future path: embeddings, finetuned transformers.  
- Guards, logging, reproducibility are universal best practices.
