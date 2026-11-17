"""Quick setup check: Gemini key + model, Tesseract OCR.  Usage: python scripts/check_setup.py"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import settings  # noqa: E402


def main() -> None:
    key = settings.gemini_api_key
    if not key or key.startswith("your-"):
        print("Gemini : NOT SET  -> put your key in .env as GEMINI_API_KEY=...")
    else:
        print(f"Gemini : key found (...{key[-4:]}, {len(key)} chars), model = {settings.gemini_model}")
        import os

        from dotenv import dotenv_values

        file_key = (dotenv_values(".env").get("GEMINI_API_KEY") or "").strip()
        if file_key and file_key != key:
            print("         warning: a GEMINI_API_KEY set in your system environment is overriding .env."
                  " Remove it (PowerShell: Remove-Item Env:GEMINI_API_KEY) or update it.")
        if key != key.strip().strip('"').strip("'"):
            print("         warning: the key has quotes or spaces around it - remove them in .env")
        try:
            from backend.llm import GeminiClient

            out = GeminiClient().generate("Reply with the single word: OK")
            print("         test call: OK ->", out.strip()[:20])
        except Exception as exc:
            print("         test call FAILED:", exc)
    from data_ingestion.pdf_parser import PDFParser

    if not settings.enable_ocr:
        print("OCR    : disabled (ENABLE_OCR=false)")
    elif PDFParser.ocr_available():
        import pytesseract

        print("OCR    : ready, tesseract", pytesseract.get_tesseract_version())
    else:
        print("OCR    : NOT available -> install Tesseract; on Windows set TESSERACT_CMD in .env")


if __name__ == "__main__":
    main()
