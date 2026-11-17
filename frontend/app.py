"""ResearchIQ Streamlit UI.  Run:  streamlit run frontend/app.py
Talks to the FastAPI backend (BACKEND_URL, default http://localhost:8000)."""
from __future__ import annotations

import os
import re

import pandas as pd
import requests
import streamlit as st

API = os.getenv("BACKEND_URL", "http://localhost:8000").rstrip("/")
st.set_page_config(page_title="ResearchIQ", page_icon="🔬", layout="wide")


def api(method: str, path: str, **kw):
    try:
        r = requests.request(method, f"{API}{path}", timeout=kw.pop("timeout", 180), **kw)
    except requests.RequestException:
        st.error("Cannot reach the ResearchIQ backend. Is it running?")
        return None
    if r.status_code >= 400:
        try:
            st.error(r.json().get("detail") or r.json().get("error") or "Request failed.")
        except ValueError:
            st.error("Request failed.")
        return None
    if r.headers.get("content-type", "").startswith("image/"):
        return r.content
    return r.json()


@st.cache_data(ttl=10)
def load_papers():
    d = api("GET", "/papers")
    return d["papers"] if d else []


def paper_label(p):
    return f"{p['paper_id']} · {p.get('short_name') or p['title'][:40]}"


def show_source_card(c: dict, key: str):
    badge = {"figure": "🖼️ Figure", "table": "📋 Table"}.get(c["chunk_type"], "📄 Text")
    pages = c["page_start"] if c["page_start"] == c["page_end"] else f"{c['page_start']}-{c['page_end']}"
    title = f"{c.get('sid') or ''} {badge} — {c['paper_id']} {c.get('short_name', '')} · {c.get('section', '')} · p.{pages}"
    with st.expander(title.strip()):
        st.caption(c.get("paper_title", ""))
        if c.get("injection_suspected"):
            st.warning("This passage contains text that looks like instructions to an AI. It was treated as data only.")
        if c.get("has_image") and c.get("element_id"):
            img = api("GET", f"/figures/{c['element_id']}/image")
            if img:
                st.image(img, caption=c["label"], use_container_width=True)
        text = c.get("text")
        if text is None:
            d = api("GET", f"/chunks/{c['chunk_id']}")
            text = d["text"] if d else ""
        st.markdown(f"<div style='font-size:0.88em;white-space:pre-wrap'>{text}</div>", unsafe_allow_html=False)
        if c.get("rerank_score") is not None:
            st.caption(f"rerank {c['rerank_score']} · dense rank {c.get('dense_rank')} · BM25 rank {c.get('sparse_rank')}")


def render_cited(items, sources):
    for it in items or []:
        sids = it.get("sources") or it.get("supporting_sources") or []
        refs = ", ".join(sources[s]["label"] for s in sids if s in sources)
        prefix = f"**{it['paper_id']}:** " if it.get("paper_id") else ""
        st.markdown(f"- {prefix}{it.get('text') or it.get('gap', '')}" + (f"  \n  <sub>{refs}</sub>" if refs else ""),
                    unsafe_allow_html=True)


st.title("🔬 ResearchIQ — Multimodal Research Assistant")
st.caption("Topic: Advanced Retrieval-Augmented Generation · answers are grounded in the indexed papers with page/figure citations")

health = api("GET", "/health")
if health:
    cols = st.columns(4)
    cols[0].metric("Papers", health["papers"])
    cols[1].metric("Indexed chunks", health["chunks"])
    cols[2].metric("LLM", "configured" if health["llm_configured"] else "missing key")
    cols[3].metric("Backend", "online")

tab_papers, tab_ask, tab_compare, tab_synth, tab_fig, tab_eval = st.tabs(
    ["📚 Papers", "💬 Ask", "⚖️ Compare", "🧭 Synthesis & gaps", "🖼️ Figures & tables", "📊 Evaluation"]
)

# ---------------------------------------------------------------- papers
with tab_papers:
    st.subheader("Upload papers (PDF)")
    ups = st.file_uploader("PDF files only, max 40 MB each", type=["pdf"], accept_multiple_files=True)
    if ups and st.button("Process uploaded papers", type="primary"):
        with st.spinner("Parsing, describing figures, chunking and indexing…"):
            res = api("POST", "/papers/upload", files=[("files", (u.name, u.getvalue(), "application/pdf")) for u in ups],
                      timeout=1800)
        if res:
            for r in res["results"]:
                (st.success if r["ok"] else st.error)(f"{r['file']}: " + ("processed" if r["ok"] else r["error"]))
            load_papers.clear()
    papers = load_papers()
    if papers:
        df = pd.DataFrame(papers)[["paper_id", "short_name", "title", "year", "source_website", "identifier", "num_pages",
                                   "num_chunks", "num_figures", "num_tables", "figures_described", "injection_flagged_chunks"]]
        st.dataframe(df, use_container_width=True, hide_index=True)
        for p in papers:
            if p.get("warnings"):
                st.warning(f"{p['paper_id']}: " + " ".join(p["warnings"]))
    else:
        st.info("No papers yet. Upload PDFs above or run `python scripts/ingest_papers.py`.")

# ---------------------------------------------------------------- ask
with tab_ask:
    papers = load_papers()
    sel = st.multiselect("Limit to papers (optional)", papers, format_func=paper_label, key="ask_sel")
    examples = [
        "What are the main research problems addressed by these papers, and how do their approaches differ?",
        "Explain the architecture shown in Figure 1 of RAG-Anything.",
        "Which limitations are explicitly acknowledged by the authors? Are any recurring?",
        "Does the collection report the energy cost of building a GraphRAG index?",
    ]
    ex = st.selectbox("Example questions", [""] + examples)
    q = st.text_area("Your question", value=ex, height=90)
    if st.button("Ask", type="primary") and q.strip():
        with st.spinner("Retrieving evidence and answering…"):
            res = api("POST", "/ask", json={"question": q, "paper_ids": [p["paper_id"] for p in sel] or None})
        if res:
            st.session_state["last_answer"] = res
    res = st.session_state.get("last_answer")
    if res:
        if res["status"] == "insufficient_evidence" or res.get("evidence_sufficient") is False:
            st.warning("⚠️ Not enough evidence in the selected papers.")
        elif res["status"] in {"llm_unavailable", "malformed_output", "blocked"}:
            st.error(res["answer"])
        for w in res.get("warnings", []):
            st.info(w)
        cits = {c["sid"]: c for c in res.get("citations", [])}
        answer = re.sub(r"\[(S\d+)\]", lambda m: f"[{cits[m.group(1)]['label']}]" if m.group(1) in cits else "", res["answer"])
        st.markdown(answer)
        if res.get("missing_information"):
            st.markdown(f"**What the papers don't tell us:** {res['missing_information']}")
        if res.get("conflicts"):
            st.markdown(f"**Conflicting evidence:** {res['conflicts']}")
        if res.get("claims"):
            with st.expander("Claim-by-claim citation check"):
                rows = [{"claim": c.get("claim"), "kind": c.get("kind"),
                         "sources": ", ".join(cits[s]["label"] for s in c.get("sources", []) if s in cits),
                         "support": c.get("support"), "score": c.get("support_score")} for c in res["claims"]]
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
                st.caption("'model_interpretation' = the assistant's own reading, not a statement from the paper. "
                           "Support score = max(lexical overlap, embedding similarity) between claim and cited passage.")
        st.subheader("Cited evidence")
        for c in res.get("citations", []):
            full = next((r for r in res["retrieved"] if r["chunk_id"] == c["chunk_id"]), c)
            show_source_card({**full, "sid": c["sid"]}, key=c["sid"])
        with st.expander(f"All retrieved passages ({len(res['retrieved'])})"):
            for i, r in enumerate(res["retrieved"]):
                show_source_card(r, key=f"r{i}")

# ---------------------------------------------------------------- compare
with tab_compare:
    papers = load_papers()
    sel = st.multiselect("Choose 2–4 papers", papers, format_func=paper_label, max_selections=4, key="cmp_sel")
    if st.button("Compare", type="primary", disabled=len(sel) < 2):
        with st.spinner("Retrieving evidence per paper and aspect…"):
            st.session_state["cmp"] = api("POST", "/compare", json={"paper_ids": [p["paper_id"] for p in sel]})
    cmp = st.session_state.get("cmp")
    if cmp:
        srcs = cmp["sources"]
        r = cmp["result"]
        rows = []
        for row in r.get("table", []):
            d = {"Aspect": row.get("aspect")}
            for pid in cmp["paper_ids"]:
                cell = (row.get("cells") or {}).get(pid, {})
                refs = "; ".join(srcs[s]["label"] for s in cell.get("sources", []) if s in srcs)
                d[pid] = f"{cell.get('text', '')}" + (f"  [{refs}]" if refs else "")
            rows.append(d)
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        for title, key in [("Similarities", "similarities"), ("Differences", "differences"), ("Trade-offs", "trade_offs"),
                           ("Limitations stated by authors", "limitations"),
                           ("Why results may not be directly comparable", "comparability_caveats")]:
            st.markdown(f"#### {title}")
            render_cited(r.get(key), srcs)
        with st.expander("Evidence used"):
            for s, c in srcs.items():
                show_source_card(c, key=f"cmp{s}")

# ---------------------------------------------------------------- synthesis
with tab_synth:
    st.caption("Literature synthesis over the whole collection (map: one summary per paper → reduce: cross-paper synthesis).")
    if st.button("Generate synthesis", type="primary"):
        with st.spinner("Summarising each paper, then synthesising…"):
            st.session_state["syn"] = api("POST", "/synthesis", json={}, timeout=900)
    syn = st.session_state.get("syn")
    if syn:
        r, srcs = syn["result"], syn["sources"]
        overview = re.sub(r"\[(S\d+)\]", lambda m: f"[{srcs[m.group(1)]['label']}]" if m.group(1) in srcs else "",
                          r.get("overview", ""))
        st.markdown("### Overview\n" + overview)
        st.markdown("### Themes")
        for t in r.get("themes", []):
            st.markdown(f"**{t.get('theme')}** — {t.get('text')}  \n<sub>{', '.join(t.get('papers', []))}</sub>",
                        unsafe_allow_html=True)
        st.markdown("### Recurring limitations")
        render_cited(r.get("recurring_limitations"), srcs)
        st.markdown("### Conflicts / inconsistencies")
        render_cited(r.get("conflicts_or_inconsistencies"), srcs)
        st.markdown("### 💡 Research gaps")
        st.info("These are AI-generated suggestions based on the evidence, not findings stated in the papers.")
        for g in r.get("research_gaps", []):
            refs = ", ".join(srcs[s]["label"] for s in g.get("supporting_sources", []) if s in srcs)
            st.markdown(f"- **{g.get('gap')}** — {g.get('why')}  \n  <sub>evidence: {refs}</sub>", unsafe_allow_html=True)
        st.markdown("### Follow-up questions (AI-generated)")
        for fq in r.get("follow_up_questions", []):
            st.markdown(f"- {fq}")
        with st.expander("Per-paper summaries"):
            st.json(syn["per_paper"])

# ---------------------------------------------------------------- figures
with tab_fig:
    papers = load_papers()
    p = st.selectbox("Paper", [None] + papers, format_func=lambda x: "All papers" if x is None else paper_label(x))
    figs = api("GET", "/figures", params={"paper_id": p["paper_id"]} if p else {}) or {"figures": []}
    figs = figs["figures"]
    st.caption(f"{len(figs)} figures/tables extracted")
    if figs:
        choice = st.selectbox("Figure / table", figs,
                              format_func=lambda f: f"{f['paper_id']} · {f['kind'].title()} {f['number']} · p.{f['page']}")
        c1, c2 = st.columns([3, 2])
        with c1:
            if choice["has_image"]:
                img = api("GET", f"/figures/{choice['element_id']}/image")
                if img:
                    st.image(img, use_container_width=True)
        with c2:
            st.markdown(f"**Caption:** {choice['caption']}")
            st.caption(f"Description source: {choice['description_source']}"
                       + (" · ⚠️ partly illegible" if choice.get("legible") is False else ""))
            fq = st.text_input("Ask about this figure/table",
                               value=f"Explain {choice['kind']} {choice['number']} of {choice['paper_id']}: what does it show?")
            if st.button("Ask about it"):
                with st.spinner("Reading the figure…"):
                    res = api("POST", "/ask", json={"question": fq, "paper_ids": [choice["paper_id"]]})
                if res:
                    cits = {c["sid"]: c for c in res.get("citations", [])}
                    st.markdown(re.sub(r"\[(S\d+)\]", lambda m: f"[{cits[m.group(1)]['label']}]" if m.group(1) in cits else "",
                                       res["answer"]))

# ---------------------------------------------------------------- evaluation dashboard
with tab_eval:
    ev = api("GET", "/evaluation/latest")
    if not ev or not ev.get("available"):
        st.info("No evaluation results yet. Run `python -m evaluation.run_eval` and refresh.")
    else:
        st.caption(f"Run at {ev['run_at']} · {ev['n_questions']} questions · K = {ev['k']} · judge: {ev.get('judge', 'n/a')}")
        st.markdown("#### Retrieval metrics (paper-level relevance labels)")
        m = ev["retrieval"]
        cols = st.columns(len(m))
        for col, (k, v) in zip(cols, m.items()):
            col.metric(k, "—" if v is None else f"{v:.3f}")
        st.markdown("#### Answer metrics")
        a = ev["answers"]
        cols = st.columns(len(a))
        for col, (k, v) in zip(cols, a.items()):
            col.metric(k, "—" if v is None else f"{v:.3f}")
        if ev.get("ragas"):
            st.markdown("#### RAGAS")
            st.bar_chart(pd.Series(ev["ragas"]))
        df = pd.DataFrame(ev["per_question"])
        st.markdown("#### By question type")
        st.dataframe(df.groupby("type")[["recall_at_k", "mrr", "correctness", "faithfulness", "citation_accuracy", "passed"]]
                     .mean(numeric_only=True).round(3), use_container_width=True)
        st.markdown("#### Per question")
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.markdown("#### Failure cases")
        for f in ev.get("failures", []):
            st.markdown(f"- **{f['id']}** ({f['type']}): {f['reason']}")
        st.caption("LLM-as-a-judge scores are indicative only and are not ground truth. Test set is small (see docs/evaluation.md).")
