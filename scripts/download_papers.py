"""Download the catalogue papers from arXiv into data/papers/ (git-ignored).

PDFs are NOT committed to the repo: arXiv's default licence does not allow redistribution.
Usage:  python scripts/download_papers.py
"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config import settings  # noqa: E402

CATALOGUE = ROOT / "docs" / "paper_catalogue.csv"


def main() -> None:
    settings.papers_dir.mkdir(parents=True, exist_ok=True)
    with CATALOGUE.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        dest = settings.papers_dir / r["filename"]
        if dest.exists():
            print(f"skip  {r['paper_id']} (already downloaded)")
            continue
        arxiv_id = r["identifier"].replace("arXiv:", "")
        url = f"https://arxiv.org/pdf/{arxiv_id}"
        print(f"get   {r['paper_id']} {url}")
        resp = requests.get(url, timeout=60, headers={"User-Agent": "ResearchIQ-hackathon/1.0"})
        resp.raise_for_status()
        if not resp.content.startswith(b"%PDF"):
            print(f"  !! {r['paper_id']}: response is not a PDF, skipped")
            continue
        dest.write_bytes(resp.content)
        time.sleep(3)  # be polite to arXiv
    print("done")


if __name__ == "__main__":
    main()
