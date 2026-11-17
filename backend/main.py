"""FastAPI backend for ResearchIQ.

Run:  uvicorn backend.main:app --reload --port 8000
Errors returned to users never contain stack traces, prompts, keys or local paths.
"""
from __future__ import annotations

import json
import logging
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from backend.guardrails import InputValidationError
from backend.llm import LLMOutputError, LLMUnavailableError
from backend.services import ResearchEngine
from config import ROOT_DIR, settings
from data_ingestion.pipeline import ingest_pdf
from data_ingestion.validators import FileValidationError, safe_filename, validate_pdf_bytes

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("researchiq")

app = FastAPI(title="ResearchIQ API", version="1.0.0")
_engine: ResearchEngine | None = None


def engine() -> ResearchEngine:
    global _engine
    if _engine is None:
        settings.ensure_dirs()
        _engine = ResearchEngine()
    return _engine


def set_engine(e: ResearchEngine) -> None:  # tests
    global _engine
    _engine = e


# ------------------------------------------------------------------ error handling
@app.exception_handler(Exception)
async def unhandled(request: Request, exc: Exception):
    rid = uuid.uuid4().hex[:8]
    log.exception("request %s failed", rid)  # full detail stays in server logs only
    return JSONResponse(status_code=500, content={"error": "Something went wrong on the server.", "request_id": rid})


def _user_error(exc: Exception) -> HTTPException:
    if isinstance(exc, (InputValidationError, FileValidationError, ValueError)):
        return HTTPException(status_code=400, detail=str(exc))
    if isinstance(exc, LLMUnavailableError):
        return HTTPException(status_code=503, detail=str(exc))
    if isinstance(exc, LLMOutputError):
        return HTTPException(status_code=502, detail="The model returned an unexpected response. Please retry.")
    raise exc


# ------------------------------------------------------------------ schemas
class AskRequest(BaseModel):
    question: str = Field(..., max_length=4000)
    paper_ids: Optional[list[str]] = None
    top_k: Optional[int] = Field(None, ge=1, le=20)


class CompareRequest(BaseModel):
    paper_ids: list[str] = Field(..., min_length=2, max_length=5)
    aspects: Optional[list[str]] = Field(None, max_length=12)


class SynthesisRequest(BaseModel):
    paper_ids: Optional[list[str]] = None


# ------------------------------------------------------------------ routes
@app.get("/health")
def health():
    e = engine()
    return {"status": "ok", "papers": len(e.store.papers), "chunks": len(e.store.chunks),
            "llm_configured": bool(getattr(e.llm, "available", False))}


@app.get("/papers")
def papers():
    return {"papers": list(engine().store.papers.values())}


@app.post("/papers/upload")
async def upload(files: list[UploadFile] = File(...)):
    results = []
    for f in files[:10]:
        name = safe_filename(f.filename or "upload.pdf")
        try:
            data = await f.read()
            validate_pdf_bytes(data, f.filename or name)
            dest = settings.papers_dir / name
            dest.write_bytes(data)
            meta = ingest_pdf(dest, engine().store)
            results.append({"file": name, "ok": True, "paper": meta})
        except FileValidationError as exc:
            results.append({"file": name, "ok": False, "error": str(exc)})
        except Exception:
            log.exception("ingest failed for %s", name)
            results.append({"file": name, "ok": False, "error": "The file could not be processed."})
    return {"results": results}


@app.delete("/papers/{paper_id}")
def delete_paper(paper_id: str):
    if paper_id not in engine().store.papers:
        raise HTTPException(404, "Unknown paper.")
    engine().store.delete_paper(paper_id)
    return {"deleted": paper_id}


@app.post("/ask")
def ask(req: AskRequest):
    try:
        return engine().ask(req.question, req.paper_ids, req.top_k)
    except Exception as exc:
        raise _user_error(exc)


@app.post("/compare")
def compare(req: CompareRequest):
    try:
        return engine().compare(req.paper_ids, req.aspects)
    except Exception as exc:
        raise _user_error(exc)


@app.post("/synthesis")
def synthesis(req: SynthesisRequest):
    try:
        return engine().synthesize(req.paper_ids)
    except Exception as exc:
        raise _user_error(exc)


@app.get("/chunks/{chunk_id}")
def chunk(chunk_id: str):
    c = engine().store.get(chunk_id)
    if c is None:
        raise HTTPException(404, "Unknown source.")
    d = c.to_dict()
    d.pop("image_path", None)  # never expose local paths
    d["has_image"] = bool(c.image_path)
    return d


@app.get("/figures")
def figures(paper_id: Optional[str] = None):
    out = []
    for c in engine().store.visuals(paper_id):
        out.append({"element_id": c.element_id, "paper_id": c.paper_id, "kind": c.chunk_type,
                    "number": c.metadata.get("number"), "page": c.page_start, "caption": c.metadata.get("caption", ""),
                    "description_source": c.metadata.get("description_source"), "legible": c.metadata.get("legible"),
                    "chunk_id": c.chunk_id, "has_image": bool(c.image_path)})
    return {"figures": sorted(out, key=lambda x: (x["paper_id"], x["page"]))}


@app.get("/figures/{element_id}/image")
def figure_image(element_id: str):
    c = engine().store.get(element_id)
    if c is None or not c.image_path:
        raise HTTPException(404, "No image for this element.")
    p = Path(c.image_path).resolve()
    if settings.figures_dir.resolve() not in p.parents or not p.exists():
        raise HTTPException(404, "No image for this element.")
    return FileResponse(p, media_type="image/png")


@app.get("/evaluation/latest")
def evaluation_latest():
    p = ROOT_DIR / "evaluation" / "results" / "latest.json"
    if not p.exists():
        return {"available": False, "message": "No evaluation run yet. Run: python -m evaluation.run_eval"}
    return {"available": True, **json.loads(p.read_text())}
