"""Shared data structures used across ingestion, retrieval and the backend."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class TextBlock:
    page: int  # 1-based page number
    text: str
    bbox: tuple[float, float, float, float]
    font_size: float = 0.0
    is_bold: bool = False
    is_heading: bool = False
    column: int = 0


@dataclass
class VisualElement:
    """A figure or a table, kept together with its caption, page and nearby text."""

    element_id: str  # e.g. P03_fig_2
    paper_id: str
    kind: str  # "figure" | "table"
    number: str  # "2", "3a" ... as printed in the caption
    page: int
    caption: str
    bbox: tuple[float, float, float, float]
    image_path: str = ""  # cropped PNG of the region
    table_markdown: str = ""  # only for tables, when extraction worked
    nearby_text: str = ""  # text right around the element on the same page
    referenced_in: list[dict[str, Any]] = field(default_factory=list)  # [{page, text}] "as shown in Figure 2..."
    description: str = ""  # model-generated visual description (clearly labelled as such)
    description_source: str = ""  # "gemini-vision" | "caption-only"
    legible: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ParsedDocument:
    paper_id: str
    title: str
    file_name: str
    num_pages: int
    blocks: list[TextBlock]
    visuals: list[VisualElement]
    warnings: list[str] = field(default_factory=list)
    ocr_pages: list[int] = field(default_factory=list)


@dataclass
class Chunk:
    chunk_id: str
    paper_id: str
    paper_title: str
    text: str
    chunk_type: str  # "text" | "figure" | "table"
    section: str
    page_start: int
    page_end: int
    element_id: str = ""  # for figure/table chunks
    figure_refs: list[str] = field(default_factory=list)  # e.g. ["Figure 2", "Table 1"] mentioned in text
    image_path: str = ""
    injection_suspected: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Chunk":
        return cls(**d)

    def citation_label(self) -> str:
        pages = f"p.{self.page_start}" if self.page_start == self.page_end else f"pp.{self.page_start}-{self.page_end}"
        if self.chunk_type in {"figure", "table"} and self.element_id:
            kind = "Figure" if self.chunk_type == "figure" else "Table"
            num = self.metadata.get("number", "")
            return f"{self.paper_id}, {kind} {num}, {pages}"
        sec = f", §{self.section}" if self.section else ""
        return f"{self.paper_id}{sec}, {pages}"
