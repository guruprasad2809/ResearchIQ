"""Layout-aware PDF parsing with PyMuPDF.

What this module does for every page:
1. Reads text blocks with font size / bold information.
2. Drops running headers, footers, page numbers and the arXiv side stamp.
3. Restores reading order for two-column layouts.
4. Finds figure and table captions, works out the region of each figure/table,
   saves a cropped PNG, and keeps caption + page + nearby text together.
5. Falls back to OCR (Tesseract) for pages without a text layer, and records a
   warning when OCR is not available instead of failing.
"""
from __future__ import annotations

import logging
import re
import statistics
from collections import Counter
from pathlib import Path

import pymupdf

from data_ingestion.schemas import ParsedDocument, TextBlock, VisualElement

log = logging.getLogger(__name__)

FIG_CAPTION_RE = re.compile(r"^\s*(Figure|Fig\.?)\s*(\d+[a-z]?)\s*[:.|]", re.IGNORECASE)
TAB_CAPTION_RE = re.compile(r"^\s*(Table|Tab\.)\s*(\d+[a-z]?)\s*[:.|]", re.IGNORECASE)
XREF_RE = re.compile(r"\b(Fig(?:ure)?s?\.?|Tables?)\s*(\d+[a-z]?)", re.IGNORECASE)
NUMBERED_HEADING_RE = re.compile(r"^(\d+(\.\d+){0,3}|[A-H](\.\d+){0,2}|[IVX]+)\.?\s+[A-Z][^.]{1,90}$")
KNOWN_HEADINGS = {
    "abstract", "introduction", "related work", "background", "method", "methods", "methodology",
    "approach", "experiments", "experimental setup", "results", "evaluation", "discussion",
    "conclusion", "conclusions", "limitations", "future work", "references", "acknowledgements",
    "acknowledgments", "appendix", "ethics statement", "broader impact",
}
ARXIV_STAMP_RE = re.compile(r"arXiv:\d{4}\.\d{4,5}(v\d+)?", re.IGNORECASE)


def _block_text(block: dict) -> tuple[str, float, bool]:
    parts, sizes, bold_chars, total_chars = [], [], 0, 0
    for line in block.get("lines", []):
        line_txt = "".join(span.get("text", "") for span in line.get("spans", []))
        parts.append(line_txt)
        for span in line.get("spans", []):
            n = len(span.get("text", "").strip())
            if n:
                sizes.append(span.get("size", 0.0))
                total_chars += n
                if span.get("flags", 0) & 16 or "bold" in span.get("font", "").lower():
                    bold_chars += n
    text = " ".join(p.strip() for p in parts if p.strip())
    text = re.sub(r"-\s+(?=[a-z])", "", text)  # join hyphenated line breaks
    text = re.sub(r"\s+", " ", text).strip()
    size = max(sizes) if sizes else 0.0
    return text, size, total_chars > 0 and bold_chars / total_chars > 0.6


def _looks_like_heading(text: str) -> bool:
    t = text.strip()
    if len(t.split()) > 14:
        return False
    plain = re.sub(r"^(\d+(\.\d+)*|[A-H]|[IVX]+)\.?\s+", "", t).strip().lower().rstrip(".:")
    return plain in KNOWN_HEADINGS or bool(NUMBERED_HEADING_RE.match(t))


def _union(rects: list[pymupdf.Rect]) -> pymupdf.Rect:
    r = pymupdf.Rect(rects[0])
    for x in rects[1:]:
        r |= x
    return r


class PDFParser:
    def __init__(self, figures_dir: Path, enable_ocr: bool = True, zoom: float = 2.0,
                 force_ocr: bool = False, ocr_text_threshold: int = 40):
        self.figures_dir = Path(figures_dir)
        self.enable_ocr = enable_ocr
        self.zoom = zoom
        self.force_ocr = force_ocr
        self.ocr_text_threshold = ocr_text_threshold

    # ------------------------------------------------------------------ public
    def parse(self, pdf_path: str | Path, paper_id: str, title: str | None = None) -> ParsedDocument:
        pdf_path = Path(pdf_path)
        doc = pymupdf.open(pdf_path)
        warnings: list[str] = []
        ocr_pages: list[int] = []
        raw_pages: list[list[TextBlock]] = []
        visuals: list[VisualElement] = []
        out_dir = self.figures_dir / paper_id
        out_dir.mkdir(parents=True, exist_ok=True)

        for pno in range(doc.page_count):
            page = doc[pno]
            try:
                blocks = self._page_blocks(page, pno + 1)
                if self.force_ocr or sum(len(b.text) for b in blocks) < self.ocr_text_threshold:
                    ocr = self._ocr_page(page, pno + 1)
                    if ocr is not None:
                        blocks, conf = ocr
                        ocr_pages.append(pno + 1)
                        if conf < 60:
                            warnings.append(
                                f"Page {pno + 1} was read with OCR at low confidence ({conf:.0f}%); "
                                "some words may be wrong."
                            )
                    elif page.get_images():
                        warnings.append(
                            f"Page {pno + 1} has no text layer and OCR is unavailable; its content is not searchable."
                        )
                page_visuals, blocks = self._extract_visuals(page, pno + 1, blocks, paper_id, out_dir)
                visuals.extend(page_visuals)
                raw_pages.append(blocks)
            except Exception as exc:  # one bad page must not kill the whole paper
                log.warning("Failed to parse page %s of %s: %s", pno + 1, pdf_path.name, exc)
                warnings.append(f"Page {pno + 1} could not be parsed and was skipped.")
                raw_pages.append([])

        blocks = self._remove_running_headers(raw_pages)
        self._mark_headings(blocks)
        self._link_references(visuals, blocks)
        visuals = self._dedupe_visuals(visuals)

        doc_title = title or (doc.metadata or {}).get("title") or self._guess_title(blocks) or pdf_path.stem
        parsed = ParsedDocument(
            paper_id=paper_id,
            title=doc_title.strip(),
            file_name=pdf_path.name,
            num_pages=doc.page_count,
            blocks=blocks,
            visuals=visuals,
            warnings=warnings,
            ocr_pages=ocr_pages,
        )
        doc.close()
        return parsed

    # ------------------------------------------------------------ page level
    def _page_blocks(self, page: pymupdf.Page, page_no: int) -> list[TextBlock]:
        W, H = page.rect.width, page.rect.height
        data = page.get_text("dict", flags=pymupdf.TEXT_PRESERVE_WHITESPACE | pymupdf.TEXT_MEDIABOX_CLIP)
        out: list[TextBlock] = []
        for b in data.get("blocks", []):
            if b.get("type") != 0:
                continue
            text, size, bold = _block_text(b)
            if not text:
                continue
            x0, y0, x1, y1 = b["bbox"]
            # arXiv side stamp: narrow vertical block on the left margin
            if ARXIV_STAMP_RE.search(text) and (x1 - x0) < W * 0.08:
                continue
            col = 0
            if x1 <= W * 0.56:
                col = 1
            elif x0 >= W * 0.44:
                col = 2
            out.append(TextBlock(page_no, text, (x0, y0, x1, y1), size, bold, False, col))
        return self._reading_order(out, H)

    @staticmethod
    def _reading_order(blocks: list[TextBlock], H: float) -> list[TextBlock]:
        """Two-column aware ordering: full-width blocks split the page into bands;
        inside a band the left column is read before the right column."""
        lefts = [b for b in blocks if b.column == 1]
        rights = [b for b in blocks if b.column == 2]
        # two-column page = some left-column block sits side by side with some right-column block
        two_col = any(min(l.bbox[3], r.bbox[3]) - max(l.bbox[1], r.bbox[1]) > 5 for l in lefts for r in rights)
        if not two_col:
            return sorted(blocks, key=lambda b: (round(b.bbox[1], 1), b.bbox[0]))
        ordered: list[TextBlock] = []
        left: list[TextBlock] = []
        right: list[TextBlock] = []

        def flush():
            ordered.extend(sorted(left, key=lambda b: b.bbox[1]))
            ordered.extend(sorted(right, key=lambda b: b.bbox[1]))
            left.clear()
            right.clear()

        for b in sorted(blocks, key=lambda b: b.bbox[1]):
            if b.column == 0:
                flush()
                ordered.append(b)
            elif b.column == 1:
                left.append(b)
            else:
                right.append(b)
        flush()
        return ordered

    @staticmethod
    def ocr_available() -> bool:
        """True when pytesseract is installed and the tesseract program can be found."""
        try:
            import pytesseract

            from config import settings

            if settings.tesseract_cmd:
                pytesseract.pytesseract.tesseract_cmd = settings.tesseract_cmd
            pytesseract.get_tesseract_version()
            return True
        except Exception:
            return False

    def _ocr_page(self, page: pymupdf.Page, page_no: int) -> tuple[list[TextBlock], float] | None:
        """OCR a page that has no text layer.

        Uses Tesseract's word boxes (image_to_data), not just plain text, so each OCR paragraph
        becomes a TextBlock with its real position and an estimated font size. That keeps the
        rest of the pipeline working on scanned pages too: two-column reading order, heading
        detection, figure/table caption detection and page-level citations.
        Returns (blocks, mean_confidence) or None when OCR is off / unavailable / found nothing.
        """
        if not self.enable_ocr:
            return None
        try:
            import pytesseract
            from PIL import Image, ImageOps
        except ImportError:
            return None
        from config import settings

        if settings.tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = settings.tesseract_cmd
        dpi = settings.ocr_dpi
        try:
            pix = page.get_pixmap(dpi=dpi)
            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            img = ImageOps.autocontrast(ImageOps.grayscale(img))  # simple clean-up for faded scans
            data = pytesseract.image_to_data(img, lang=settings.ocr_lang, output_type=pytesseract.Output.DICT)
        except Exception as exc:  # tesseract binary missing, language pack missing, etc.
            log.info("OCR failed on page %s: %s", page_no, exc)
            return None

        scale = 72.0 / dpi  # image pixels -> PDF points
        # 1) collect OCR lines (word boxes grouped by Tesseract's block/paragraph/line ids)
        lines: dict[tuple, dict] = {}
        confs: list[float] = []
        for i, word in enumerate(data["text"]):
            word = (word or "").strip()
            try:
                conf = float(data["conf"][i])
            except (TypeError, ValueError):
                conf = -1
            if not word or conf < settings.ocr_min_conf:
                continue
            confs.append(conf)
            key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
            x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
            ln = lines.setdefault(key, {"words": [], "box": [x, y, x + w, y + h], "heights": []})
            ln["words"].append(word)
            b = ln["box"]
            ln["box"] = [min(b[0], x), min(b[1], y), max(b[2], x + w), max(b[3], y + h)]
            ln["heights"].append(h)
        if not lines:
            return None

        # 2) merge lines into paragraphs, but start a new block at a heading, a caption,
        #    or a clear font-size change (Tesseract often glues these to the next paragraph)
        groups: list[list[dict]] = []
        prev_par = None
        for key in sorted(lines):
            ln = lines[key]
            ln["text"] = " ".join(ln["words"])
            ln["size"] = statistics.median(ln["heights"])
            par = key[:2]
            starts_new = (
                par != prev_par
                or not groups
                or _looks_like_heading(ln["text"])
                or FIG_CAPTION_RE.match(ln["text"]) or TAB_CAPTION_RE.match(ln["text"])
                or _looks_like_heading(groups[-1][-1]["text"])  # a heading line closes its block
                or (  # a caption ends at its full stop; the table header after it is a new block
                    (FIG_CAPTION_RE.match(groups[-1][0]["text"]) or TAB_CAPTION_RE.match(groups[-1][0]["text"]))
                    and groups[-1][-1]["text"].rstrip().endswith(".")
                )
                or abs(ln["size"] - groups[-1][-1]["size"]) > 0.25 * max(ln["size"], groups[-1][-1]["size"])
            )
            if starts_new:
                groups.append([ln])
            else:
                groups[-1].append(ln)
            prev_par = par

        W = page.rect.width
        blocks: list[TextBlock] = []
        for g in groups:
            text = " ".join(ln["text"] for ln in g)
            text = re.sub(r"-\s+(?=[a-z])", "", text)
            text = re.sub(r"\s+", " ", text).strip()
            if not text:
                continue
            x0 = min(ln["box"][0] for ln in g) * scale
            y0 = min(ln["box"][1] for ln in g) * scale
            x1 = max(ln["box"][2] for ln in g) * scale
            y1 = max(ln["box"][3] for ln in g) * scale
            # word-box height ~ font size in points (only the ratio to body text matters)
            size = statistics.median([h for ln in g for h in ln["heights"]]) * scale * 1.1
            col = 1 if x1 <= W * 0.56 else 2 if x0 >= W * 0.44 else 0
            blocks.append(TextBlock(page_no, text, (x0, y0, x1, y1), round(size, 1), False, False, col))
        mean_conf = sum(confs) / len(confs) if confs else 0.0
        return self._reading_order(blocks, page.rect.height), mean_conf

    # ------------------------------------------------------------ visuals
    def _graphic_rects(self, page: pymupdf.Page) -> list[pymupdf.Rect]:
        rects: list[pymupdf.Rect] = []
        for info in page.get_image_info():
            r = pymupdf.Rect(info["bbox"])
            if r.width > 30 and r.height > 30:
                rects.append(r)
        try:
            drawings = page.get_drawings()
        except Exception:
            drawings = []
        for d in drawings:
            r = pymupdf.Rect(d.get("rect"))
            if r.width > 4 or r.height > 4:
                rects.append(r)
        return rects

    def _extract_visuals(self, page, page_no, blocks, paper_id, out_dir):
        W, H = page.rect.width, page.rect.height
        captions = []
        for i, b in enumerate(blocks):
            m = FIG_CAPTION_RE.match(b.text)
            if m:
                captions.append(("figure", m.group(2), i, b))
                continue
            m = TAB_CAPTION_RE.match(b.text)
            if m:
                captions.append(("table", m.group(2), i, b))
        if not captions:
            return [], blocks

        graphics = self._graphic_rects(page)
        try:
            tables = list(page.find_tables().tables)
        except Exception:
            tables = []

        visuals: list[VisualElement] = []
        consumed: set[int] = set()
        for kind, number, idx, cap in captions:
            cap_rect = pymupdf.Rect(cap.bbox)
            consumed.add(idx)
            table_md = ""
            if kind == "figure":
                region = self._figure_region(cap_rect, graphics, blocks, W, H)
            else:
                region, table_md = self._table_region(cap_rect, tables, blocks, W, H)
            region = region & page.rect
            # text blocks that sit inside the graphic (axis labels, legend, node names)
            inside = [
                j for j, b in enumerate(blocks)
                if j != idx and pymupdf.Rect(b.bbox).intersects(region)
                and (pymupdf.Rect(b.bbox) & region).get_area() > 0.6 * pymupdf.Rect(b.bbox).get_area()
                and not FIG_CAPTION_RE.match(b.text) and not TAB_CAPTION_RE.match(b.text)
            ]
            # do not swallow long paragraphs or section headings: those are body text, not labels
            labels = [j for j in inside if len(blocks[j].text.split()) <= 25 and not _looks_like_heading(blocks[j].text)]
            consumed.update(labels)
            label_text = " | ".join(blocks[j].text for j in labels)[:1500]

            crop = region | cap_rect
            image_path = ""
            if crop.width > 20 and crop.height > 20:
                element_id = f"{paper_id}_{kind[:3]}_{number}"
                image_path = str(out_dir / f"{element_id}_p{page_no}.png")
                try:
                    page.get_pixmap(matrix=pymupdf.Matrix(self.zoom, self.zoom), clip=crop).save(image_path)
                except Exception as exc:
                    log.info("Could not render %s: %s", element_id, exc)
                    image_path = ""

            nearby = self._nearby_text(blocks, idx, consumed)
            if label_text:
                nearby = (nearby + "\nLabels inside the " + kind + ": " + label_text).strip()
            visuals.append(
                VisualElement(
                    element_id=f"{paper_id}_{kind[:3]}_{number}",
                    paper_id=paper_id,
                    kind=kind,
                    number=number,
                    page=page_no,
                    caption=cap.text[:1200],
                    bbox=tuple(round(v, 1) for v in crop),
                    image_path=image_path,
                    table_markdown=table_md,
                    nearby_text=nearby[:1500],
                )
            )
        remaining = [b for j, b in enumerate(blocks) if j not in consumed]
        return visuals, remaining

    @staticmethod
    def _same_column(r: pymupdf.Rect, cap: pymupdf.Rect, W: float) -> bool:
        if cap.width > W * 0.6:  # full-width caption -> any column
            return True
        return min(r.x1, cap.x1) - max(r.x0, cap.x0) > -10

    def _figure_region(self, cap, graphics, blocks, W, H) -> pymupdf.Rect:
        # figures normally sit directly above their caption
        cands = [
            g for g in graphics
            if g.y1 <= cap.y0 + 8 and g.y0 >= cap.y0 - H * 0.65 and self._same_column(g, cap, W)
        ]
        if cands:
            region = _union(cands)
            # stop at the nearest long paragraph above the caption so we don't grab body text
            paras_above = [
                pymupdf.Rect(b.bbox) for b in blocks
                if b.bbox[3] <= region.y0 + 2 and len(b.text.split()) > 40 and self._same_column(pymupdf.Rect(b.bbox), cap, W)
            ]
            if paras_above:
                top = max(p.y1 for p in paras_above)
                region.y0 = max(region.y0, top)
            return region
        # no vector/raster graphics detected: take the empty gap above the caption
        above = [
            pymupdf.Rect(b.bbox) for b in blocks
            if b.bbox[3] <= cap.y0 - 2 and len(b.text.split()) > 25 and self._same_column(pymupdf.Rect(b.bbox), cap, W)
        ]
        top = max((r.y1 for r in above), default=max(0, cap.y0 - H * 0.35))
        x0, x1 = (cap.x0, cap.x1) if cap.width < W * 0.6 else (0, W)
        return pymupdf.Rect(x0, top, x1, cap.y0)

    def _table_region(self, cap, tables, blocks, W, H) -> tuple[pymupdf.Rect, str]:
        best, best_dist = None, 1e9
        for t in tables:
            r = pymupdf.Rect(t.bbox)
            if not self._same_column(r, cap, W):
                continue
            dist = r.y0 - cap.y1 if r.y0 >= cap.y1 - 5 else cap.y0 - r.y1  # below or above caption
            if -5 <= dist < best_dist and dist < H * 0.3:
                best, best_dist = t, dist
        if best is not None:
            try:
                md = best.to_markdown()
            except Exception:
                md = ""
            return pymupdf.Rect(best.bbox), md[:6000]
        # fallback: table captions usually sit above the table; stop at the next paragraph, heading or caption
        below = [
            pymupdf.Rect(b.bbox) for b in blocks
            if b.bbox[1] >= cap.y1 + 2 and self._same_column(pymupdf.Rect(b.bbox), cap, W)
            and (len(b.text.split()) > 40 or _looks_like_heading(b.text)
                 or FIG_CAPTION_RE.match(b.text) or TAB_CAPTION_RE.match(b.text))
        ]
        bottom = min((r.y0 for r in below), default=min(H, cap.y1 + H * 0.35))
        x0, x1 = (cap.x0, cap.x1) if cap.width < W * 0.6 else (0, W)
        return pymupdf.Rect(x0, cap.y1, x1, bottom), ""

    @staticmethod
    def _nearby_text(blocks, idx, consumed, window: int = 2) -> str:
        parts = []
        for j in range(max(0, idx - window), min(len(blocks), idx + window + 1)):
            if j == idx or j in consumed:
                continue
            if len(blocks[j].text.split()) >= 8:
                parts.append(blocks[j].text)
        return " ".join(parts)[:1200]

    # ------------------------------------------------------------ document level
    @staticmethod
    def _remove_running_headers(pages: list[list[TextBlock]]) -> list[TextBlock]:
        counter: Counter[str] = Counter()
        for blocks in pages:
            seen = set()
            for b in blocks:
                key = re.sub(r"\d+", "#", b.text.lower())[:80]
                if len(b.text.split()) <= 14 and key not in seen:
                    counter[key] += 1
                    seen.add(key)
        n_pages = max(1, len(pages))
        repeated = {k for k, c in counter.items() if n_pages >= 3 and c >= max(3, int(0.5 * n_pages))}
        out: list[TextBlock] = []
        for blocks in pages:
            for b in blocks:
                key = re.sub(r"\d+", "#", b.text.lower())[:80]
                if key in repeated:
                    continue
                if re.fullmatch(r"\d{1,3}", b.text.strip()):  # bare page number
                    continue
                out.append(b)
        return out

    @staticmethod
    def _mark_headings(blocks: list[TextBlock]) -> None:
        sizes = [b.font_size for b in blocks if b.font_size > 0 and len(b.text.split()) > 15]
        body = statistics.median(sizes) if sizes else 10.0
        for b in blocks:
            words = b.text.split()
            if not words or len(words) > 16 or FIG_CAPTION_RE.match(b.text) or TAB_CAPTION_RE.match(b.text):
                continue
            plain = re.sub(r"^(\d+(\.\d+)*|[A-H]|[IVX]+)\.?\s+", "", b.text).strip().lower().rstrip(".:")
            if plain in KNOWN_HEADINGS:
                b.is_heading = True
            elif b.font_size >= body * 1.12 and len(words) <= 14 and not b.text.endswith("."):
                b.is_heading = True
            elif b.is_bold and NUMBERED_HEADING_RE.match(b.text):
                b.is_heading = True

    @staticmethod
    def _link_references(visuals: list[VisualElement], blocks: list[TextBlock]) -> None:
        index = {(v.kind, v.number.lower()): v for v in visuals}
        for b in blocks:
            for m in XREF_RE.finditer(b.text):
                kind = "table" if m.group(1).lower().startswith("tab") else "figure"
                v = index.get((kind, m.group(2).lower()))
                if v is None or len(v.referenced_in) >= 3:
                    continue
                start = max(0, m.start() - 200)
                snippet = b.text[start : m.end() + 250]
                v.referenced_in.append({"page": b.page, "text": snippet})

    @staticmethod
    def _dedupe_visuals(visuals: list[VisualElement]) -> list[VisualElement]:
        seen: dict[str, VisualElement] = {}
        for v in visuals:
            if v.element_id in seen:  # e.g. "Figure 2:" repeated in appendix -> keep both with suffix
                v.element_id = f"{v.element_id}_p{v.page}"
            seen[v.element_id] = v
        return list(seen.values())

    @staticmethod
    def _guess_title(blocks: list[TextBlock]) -> str:
        first = [b for b in blocks if b.page == 1 and 2 <= len(b.text.split()) <= 25]
        if not first:
            return ""
        return max(first, key=lambda b: b.font_size).text
