from data_retrieval.hybrid_search import HybridRetriever, resolve_papers, rrf_fuse
from evaluation.metrics import key_fact_coverage, retrieval_metrics


def test_rrf_fusion_rewards_agreement():
    s = rrf_fuse([["a", "b", "c"], ["b", "a", "d"]], k=60)
    assert s["a"] == s["b"] and s["a"] > s["c"] and s["c"] == s["d"]


def test_hybrid_search_finds_right_paper(loaded_store):
    r = HybridRetriever(loaded_store)
    hits = r.search("How does GraphWeave summarise communities of entities?", top_k=5)
    assert hits and hits[0].chunk.paper_id == "P01"
    hits = r.search("DenseFlow limitations", top_k=5, paper_ids=["P02"])
    assert hits and all(h.chunk.paper_id == "P02" for h in hits)


def test_explicit_figure_reference_is_pinned(loaded_store):
    hits = HybridRetriever(loaded_store).search("Explain Figure 1 of DenseFlow", top_k=5)
    assert hits[0].pinned and hits[0].chunk.element_id == "P02_fig_1"


def test_resolve_papers_by_alias(loaded_store):
    assert set(resolve_papers("compare graphweave and DenseFlow", loaded_store.papers)) == {"P01", "P02"}


def test_store_persists_and_deletes(loaded_store):
    n = len(loaded_store.chunks)
    loaded_store.delete_paper("P02")
    assert all(c.paper_id != "P02" for c in loaded_store.chunks.values()) and len(loaded_store.chunks) < n
    assert "P02" not in loaded_store.papers


def test_retrieval_metrics_known_values():
    hits = [{"paper_id": p, "page_start": 1, "page_end": 1} for p in ["X", "A", "B", "A"]]
    m = retrieval_metrics(hits, ["A", "C"], k=4)
    assert m["recall_at_k"] == 0.5
    assert m["precision_at_k"] == 0.5
    assert m["mrr"] == 0.5
    assert 0 < m["ndcg_at_k"] < 1


def test_page_level_relevance():
    hits = [{"paper_id": "A", "page_start": 3, "page_end": 4}]
    assert retrieval_metrics(hits, ["A"], 1, {"A": [4]})["mrr"] == 1.0
    assert retrieval_metrics(hits, ["A"], 1, {"A": [9]})["mrr"] == 0.0


def test_key_fact_coverage():
    assert key_fact_coverage("It uses PPO and GRPO with an outcome-based reward.", ["PPO", "GRPO", "outcome-based reward"]) == 1.0
    assert key_fact_coverage("No idea.", ["PPO"]) == 0.0
