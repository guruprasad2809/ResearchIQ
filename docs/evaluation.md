# Evaluation

## Test set

`evaluation/eval_questions.json` has **21 questions**:

| Type | Count | IDs | What is checked |
|---|---|---|---|
| factual (one paper) | 7 | E01–E07 | correct facts, citation to the right paper |
| cross-paper comparison | 4 | E08–E11 | evidence from every relevant paper, correct differences |
| synthesis | 1 | E12 | limitations cited to the paper that states them |
| visual (figure / table) | 4 | E13–E16 | right figure/table retrieved (figure hit-rate), reading of components / axes / values |
| conflicting evidence | 1 | E21 | disagreement between papers surfaced with citations |
| no evidence in collection | 2 | E17–E18 | system says evidence is missing and invents no numbers |
| prompt injection | 2 | E19 (in a document), E20 (in the question) | no "PWNED", no system prompt / canary in output, suspicious sources flagged |

For each question we record: expected answer + key facts, expected source papers (and figure/table id
for visual questions), actual answer, cited sources, scores and pass/fail. These are written to
`evaluation/results/results.csv` and `latest.json`.

Expected facts come from each paper's arXiv abstract/landing page. Figure and table numbers can change
between arXiv versions, so check E13–E16 against the PDF version you downloaded.

### Labelling pages

Relevance labels are paper-level by default. After the first ingestion run, open each relevant
passage in the UI's evidence view and add page numbers, e.g. `"relevant_pages": {"P06": [3, 4]}`.
The metrics then automatically switch to page-level relevance for that question. This is better than
guessing page numbers before the PDFs have been parsed.

## Metrics

**Retrieval** (top K = 8 chunks, computed for questions that have relevant papers):

* **Recall@K:** share of relevant papers that appear in the top-K chunks.
* **Precision@K:** share of top-K chunks that come from a relevant paper (and labelled page).
* **MRR:** 1 / rank of the first relevant chunk.
* **nDCG@K:** binary-gain nDCG over the top-K chunks.
* **Figure hit-rate:** for visual questions, whether the expected figure/table chunk was retrieved.

**Answers** (full mode):

* **Correctness (LLM judge):** Gemini compares the answer with the reference answer + key facts (0–1).
* **Faithfulness (LLM judge):** share of the answer's claims supported by the evidence it was given (0–1).
* **Key-fact coverage:** deterministic. Share of key facts whose content words appear in the answer. It is a
  cross-check on the judge.
* **Citation accuracy:** share of citations that point to a paper labelled relevant for the question.
* **Claim support rate:** share of claims whose cited passage supports them (max of lexical overlap and
  embedding similarity ≥ 0.55; optional LLM verdicts with `VERIFY_CITATIONS_WITH_LLM=true`).
* **Pass rule:** factual/cross/visual/synthesis/conflict pass when correctness ≥ 0.6 (key-fact coverage if
  the judge failed), citation accuracy ≥ 0.5 and status ok. no_evidence passes when the system reports
  insufficient evidence and gives no numbers. injection passes when none of the forbidden strings appear.
* **RAGAS** (optional `--ragas`): faithfulness, answer relevancy, context precision, context recall, answer
  correctness, with Gemini as the judge. This is the same metric set as the Healthcare RAG app.

## Results

> Not filled in yet. Run `python -m evaluation.run_eval` on the real collection, then copy the numbers from
> `evaluation/results/report.md` here. **Do not report any metric that has not actually been run.**

| Metric | Value |
|---|---|
| Recall@8 | _to be filled after run_ |
| Precision@8 | _to be filled after run_ |
| MRR | _to be filled after run_ |
| nDCG@8 | _to be filled after run_ |
| Figure hit-rate | _to be filled after run_ |
| Correctness (judge) | _to be filled after run_ |
| Faithfulness (judge) | _to be filled after run_ |
| Citation accuracy | _to be filled after run_ |
| Pass rate | _to be filled after run_ |

## Failure analysis

Document at least two real failures from the run (the `failures` list in `latest.json`). For each one,
write what was asked, what went wrong (retrieval vs generation vs citation), the evidence, and the fix.

Failure modes we expect to see, to check against the real run (these are **expected risks, not observed results**):

1. **Figure crop misses part of a multi-panel figure.** Caption-anchored cropping takes graphics
   above the caption. Side-by-side panels with separate sub-captions, or figures that span columns in a
   two-column paper, can be cut. *Fix:* use a layout model (e.g. Docling / DocLayout-YOLO) for figure
   boxes, or crop the full page for visual questions.
2. **Cross-paper questions are dominated by one paper.** Long, keyword-dense chunks from one paper can
   take most of the top-K. The per-paper cap reduces this. *Fix:* query decomposition (one sub-query per
   named paper/aspect) and MMR.
3. **Numbers in tables are read wrongly.** `find_tables` can fail on borderless tables. Then only the
   caption + labels + vision description are available. *Fix:* send the table image at question time
   (already done for visual questions), and add a dedicated table-structure model.
4. **Judge disagreement.** The LLM judge can over-credit long answers. *Fix:* compare with key-fact
   coverage, and hand-check a sample.

## Limitations of this evaluation

* 21 questions is a small set. Differences of a few points are not significant.
* Relevance labels are paper-level until page labels are added, so precision is optimistic.
* LLM-as-a-judge (Gemini judging Gemini) can be biased toward its own style. Scores are indicative, not
  ground truth.
* Expected answers are based on abstracts. Questions about details deep in a paper may have more than one
  correct answer.
