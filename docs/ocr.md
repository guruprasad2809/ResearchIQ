# OCR for scanned pages

ResearchIQ reads normal PDFs directly with PyMuPDF. A page with almost no text (fewer than
`OCR_TEXT_THRESHOLD` = 40 characters), usually a scanned page, is read with **Tesseract OCR**.

## How it works

1. PyMuPDF tries to read the page text. If there is enough text, OCR is skipped.
2. Otherwise PyMuPDF renders the page as an image at `OCR_DPI` (300 by default).
3. Pillow converts it to greyscale and stretches the contrast (helps faded scans).
4. `pytesseract.image_to_data` runs Tesseract and returns **every word with its box and a confidence score**.
5. Words below `OCR_MIN_CONF` (40) are dropped.
6. Words are grouped into lines, and lines into blocks. A new block starts at a heading, a
   `Figure N:` / `Table N:` caption, or where the font size changes. Tesseract often glues these to the next paragraph.
7. Each block gets its real position (pixels → PDF points) and an estimated font size (from word height).
8. From here the page goes through **the same pipeline as a normal page**: two-column reading order, heading
   detection, figure/table caption detection and cropping, chunking, and page-level citations.
9. The page number is saved in the paper's `ocr_pages`. If mean confidence is below 60%, the paper gets a
   warning that some words may be wrong.
10. If Tesseract is not installed, nothing crashes. The page gets a warning that it is not searchable.

## Install

| OS | Command |
|---|---|
| Windows | Install from the UB Mannheim Tesseract page (`tesseract-ocr-w64-setup-*.exe`). Default path: `C:\Program Files\Tesseract-OCR\tesseract.exe` |
| macOS | `brew install tesseract` |
| Ubuntu/Debian | `sudo apt install tesseract-ocr` |

Python side (already in `requirements.txt`): `pip install pytesseract pillow`.

Check: `tesseract --version`. On Windows, if that fails, set the full path in `.env`:

```
TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe
```

Other languages: install the language pack (e.g. `sudo apt install tesseract-ocr-deu`) and set `OCR_LANG=eng+deu`.

## Settings (`.env`)

| Setting | Default | Meaning |
|---|---|---|
| `ENABLE_OCR` | true | turn OCR on/off |
| `TESSERACT_CMD` | (empty) | full path to tesseract if it is not on PATH |
| `OCR_LANG` | eng | Tesseract language(s) |
| `OCR_DPI` | 300 | render resolution; 200 is faster, 300 more accurate |
| `OCR_MIN_CONF` | 40 | drop words below this confidence (0–100) |
| `OCR_TEXT_THRESHOLD` | 40 | pages with fewer characters are treated as scanned |
| `OCR_FORCE` | false | OCR every page, even with a text layer (testing / broken text layers) |

## Test and demo

```
pytest -q tests/test_ocr.py                                  # 5 OCR tests (skipped if Tesseract is missing)
python scripts/make_scanned_pdf.py data/papers/P05_lightrag.pdf --pages 1-4
python scripts/ingest_papers.py --with-scanned P05           # ingests an image-only copy as paper S05
```

Then in the app: the top bar shows **OCR: ready**, and the Papers tab shows S05 with `ocr_pages` 1, 2, 3, 4.
Ask "In the scanned copy of LightRAG, what is the main idea?". The citations point to S05 with page numbers.

## Limits

* Maths, tiny fonts and complex tables come out messy.
* Bold is not detected on OCR pages. Headings are found by size and numbering only.
* About 1–3 s per page at 300 DPI.
* Possible upgrades: deskewing, PaddleOCR/Docling for tables, or Gemini Vision as the OCR for hard pages.
