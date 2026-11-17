# Design decisions, limitations, risks and improvements

## Key decisions

| Decision | Why | Alternative considered |
|---|---|---|
| PyMuPDF for parsing | Fast, no GPU. Gives bounding boxes, fonts, images, vector drawings, tables and region rendering in one library. | Docling / Unstructured: better layout models, but slower and heavier to install in a one-day hackathon. |
| Caption-anchored figure cropping | Captions are very reliable in papers ("Figure 3:"). Anchoring on them links figure ↔ caption ↔ page with no extra model. | Layout detection model (more accurate boxes, more setup). |
| Semantic title-based chunks with `[paper | section]` header | Chunks stay on one topic, citations get an exact section, and the header helps both BM25 and dense retrieval know the source. | Fixed-size token windows (simpler, but mix sections and lose context). |
| Hybrid BM25 + dense with RRF | Paper questions contain exact terms (model names, benchmarks, "GRPO") that BM25 catches, and paraphrases that embeddings catch. RRF needs no score tuning. | Weighted score sum (needs normalisation and tuning). |
| Cross-encoder rerank | Cheap precision boost, and its score gives a clear "no relevant evidence" threshold. | LLM reranking (slower, costs tokens). |
| JSON answers with claims + source ids | Makes citations checkable in code (whitelist + support score) instead of trusting free text. | Free-text answers with regex-parsed citations. |
| Separate visual chunks | Figures and tables become first-class search results with their own citation (`P06, Figure 1, p.3`). | Only page-level text (can't answer "what does Figure 3 show"). |

## The main engineering trade-off: describe figures once vs read them at question time

* **Describe at ingestion:** each figure is described once, the description is indexed, and retrieval
  can find a figure from what it *shows* (e.g. "chart of recall vs chunk size") even if the caption
  doesn't say it. The cost is one vision call per figure at ingest, and a fixed description that may
  miss what a later question needs.
* **Read at question time:** the model sees the real pixels for the exact question, so it is more
  accurate for axes, values and trends. The cost is latency and tokens on every visual question.

We do **both**. Descriptions make figures findable. For visual questions, the top 2 figure/table
crops are also sent with the question. Facts the model reads off an image are labelled
`model_interpretation` unless the caption or paper text says the same thing.

## How we verify that a citation supports a claim

1. The model must return each claim with its source ids. Ids it wasn't given are removed.
2. Each (claim, cited passage) pair gets a support score = max(lexical content-word recall, embedding
   cosine). It is labelled supported (≥ 0.55), weak (≥ 0.35) or unsupported. These labels are shown
   in the UI's claim table.
3. Optionally (`VERIFY_CITATIONS_WITH_LLM=true`) an LLM judge gives supported / partially / not
   supported for each pair.
4. In evaluation, citation accuracy also checks that cited papers are the labelled relevant ones.

## What happens if the LLM provider is down or returns bad output

Retries with backoff happen first. If it is still down, `/ask` returns status `llm_unavailable` with
the retrieved passages, so the user still gets evidence. Bad JSON is repaired once. If that fails, the
status is `malformed_output`. Comparison/synthesis return a 503/502 with a short message. Stack traces
stay in the logs.

## Scaling to thousands of papers

* Move BM25 into Qdrant sparse vectors (or OpenSearch). In-memory BM25 is rebuilt on every change.
* Run Qdrant as a server with payload indexes on `paper_id`, `chunk_type`, `year`. Use quantisation for
  memory.
* Make ingestion a background job queue (Celery/RQ) with per-paper status, and batch embeddings on GPU.
* Add a paper-level index (title/abstract/summary vectors): first pick candidate papers, then search
  chunks inside them (two-stage retrieval).
* Pre-compute and cache per-paper summaries for synthesis. Synthesise over clusters of papers
  (GraphRAG-style communities) instead of all papers at once.

## Known limitations and risks

* Figure crops can be wrong for multi-panel or column-spanning figures, and for figures above a page break.
* `find_tables` misses some borderless tables. Then table numbers depend on the vision model.
* The injection detector is pattern-based. A cleverly paraphrased injection can pass detection. Spotlighting,
  the system rules and the canary are the second line of defence, but no defence is complete.
* Heading detection is heuristic. Unusual templates can produce coarse sections (citations still have pages).
* LLM-judge metrics are not ground truth. The test set is small.
* Local Qdrant file mode allows only one process at a time.
* Model-generated figure descriptions can be wrong. They are labelled, and the image is always viewable.

## Proposed improvements

1. Layout model (Docling or DocLayout-YOLO) for figure/table boxes, and equations as LaTeX.
2. Query decomposition + MMR for multi-paper questions. HyDE-style expansion for vague questions.
3. ColPali-style page-image retrieval as a third retriever for visually dense pages.
4. Claim-level NLI model for citation verification instead of lexical/cosine heuristics.
5. Classifier-based injection detection (e.g. a small fine-tuned model) in addition to patterns.
6. Page-level relevance labels for every eval question, and a bigger test set with two annotators.
