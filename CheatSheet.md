# Week-1 RAG — One-Page Cheat Sheet (Line → Knob → Failure → Test)

| Line / Concept | Main Knob(s) | Typical Failure Mode | Quick Test |
|---|---|---|---|
| `RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)` | `chunk_size`, `chunk_overlap` | Too big → truncation/noisy retrieval; too small → context fragmentation | Sweep sizes (400/60, 800/120, 1000/150) and track EM/Accuracy change |
| `OpenAIEmbeddings(model="text-embedding-3-small")` | embedding model | Mixed models across index/query → bad recall | Embed a fixed probe set, verify nearest neighbors stay stable across runs |
| `FAISS.from_documents(...); db.save_local()/load_local()` | persist path, re-use index | Re-embedding every run → slow, non-repro | Time a prompt-sweep with and without `load_local()` |
| `retriever = db.as_retriever(search_kwargs={"k":4})` | `k` (evidence bandwidth) | k too low → miss context; k too high → noise/hallucination | Sweep k = 2,4,6; pick best EM/F1 |
| `PromptTemplate(... "Use the context...")` | instruction phrasing | LLM cites outside context; verbose answers | Add “Only use context; say ‘I don’t know’ otherwise” and measure |
| `ChatOpenAI(..., temperature=0)` | `temperature` | Non-deterministic eval; style variance masks regressions | Fix temp=0 for eval; allow higher temp only in UX demos |
| `RetrievalQA.from_chain_type(..., chain_type="stuff")` | chain type (`stuff`/`map_reduce`/`refine`) | Latency high or context overrun on long docs | Start with `stuff`; move to `map_reduce` for long contexts |
| TF-IDF + `PassiveAggressiveClassifier` (Day-4) | `C`, n-grams | Over/under-fit; unfair RAG comparison | 5-fold CV on headline dataset; compare to RAG EM on same set |
| Milvus `connections.connect(...)` | host/port/alias | Silent connection fail; hitting wrong cluster | `list_collections()`; assert `week1_rag` exists before queries |
| `Milvus.from_documents(..., collection_name="week1_rag")` | collection name | Orphaned/mixed collections | Version names: `week1_rag_v1`, `_v2`; migration script |
| `retriever = milvus.as_retriever({"k":5})` | `k` | Same as FAISS; scale changes optimal k | Re-run k-sweep after moving to Milvus |
| Eval loop `exact_match(y_true, y_pred)` | metric choice | EM too strict for paraphrases | Track EM + ROUGE-L; manual spot-check 10 samples |
| `make_chunks / build_store / make_qa` | function seams | Tight coupling; hard swaps | Unit-test each func; inject backends/prompts via args |
| `ask(q)` entrypoint | logging, caching | No single hook for telemetry/guardrails | Add timing, trace IDs; optional cache with hash(context+q) |
| `safe_ask` try/except | fallback message | Demo crash on transient errors | Force an exception (mock retriever) and confirm graceful fallback |
| Gradio `Interface` | inputs/outputs | UI mismatch (list vs string); encoding issues | Smoke test with 5 typical and 3 edge-case queries |
