"""Main document parsing pipeline wrapping the GLM-OCR SDK."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tqdm import tqdm

from doc_parser.config import get_settings
from doc_parser.post_processor import assemble_markdown, save_to_json

logger = logging.getLogger(__name__)

try:
    from glmocr import GlmOcr  # type: ignore[import]
    _GLMOCR_AVAILABLE = True
except ImportError:
    _GLMOCR_AVAILABLE = False
    logger.warning(
        "glmocr package not installed. Install with: uv pip install glmocr"
    )


@dataclass
class ParsedElement:
    """A single detected and recognized document element.

    Attributes:
        label: Element category (e.g. 'document_title', 'paragraph', 'table').
        text: Recognized text content.
        bbox: Bounding box [x1, y1, x2, y2] in normalized coordinates.
        score: Detection confidence score (0.0–1.0).
        reading_order: Sequence index for correct reading order assembly.
    """

    label: str
    text: str
    bbox: list[float]
    score: float
    reading_order: int


@dataclass
class PageResult:
    """Parsing result for a single document page.

    Attributes:
        page_num: One-based page number.
        elements: All detected elements on this page.
        markdown: Assembled Markdown string for this page.
    """

    page_num: int
    elements: list[ParsedElement] = field(default_factory=list)
    markdown: str = ""


@dataclass
class ParseResult:
    """Complete parsing result for a document.

    Attributes:
        source_file: Path to the original document.
        pages: Per-page parsing results.
        total_elements: Sum of all elements across all pages.
    """

    source_file: str
    pages: list[PageResult] = field(default_factory=list)
    total_elements: int = 0

    @classmethod
    def from_sdk_result(cls, raw: Any, source_file: str) -> ParseResult:
        """Build a ParseResult from a raw GLM-OCR SDK response.

        The SDK response structure is expected to be a dict or object with
        page-level results, each containing a list of element dicts with
        'label', 'text', 'bbox', 'score', and optionally 'reading_order'.

        Args:
            raw: Raw response from the GlmOcr.parse() call.
            source_file: Path to the source document.

        Returns:
            Populated ParseResult instance.
        """
        pages: list[PageResult] = []

        # Handle both dict-like and attribute-based SDK responses
        raw_pages = raw if isinstance(raw, list) else getattr(raw, "pages", [raw])

        for page_idx, raw_page in enumerate(raw_pages):
            page_num = page_idx + 1
            raw_elements = (
                raw_page.get("elements", [])
                if isinstance(raw_page, dict)
                else getattr(raw_page, "elements", [])
            )

            elements: list[ParsedElement] = []
            for order_idx, raw_el in enumerate(raw_elements):
                if isinstance(raw_el, dict):
                    el = ParsedElement(
                        label=raw_el.get("label", "paragraph"),
                        text=raw_el.get("text", ""),
                        bbox=raw_el.get("bbox", [0.0, 0.0, 1.0, 1.0]),
                        score=raw_el.get("score", 1.0),
                        reading_order=raw_el.get("reading_order", order_idx),
                    )
                else:
                    el = ParsedElement(
                        label=getattr(raw_el, "label", "paragraph"),
                        text=getattr(raw_el, "text", ""),
                        bbox=getattr(raw_el, "bbox", [0.0, 0.0, 1.0, 1.0]),
                        score=getattr(raw_el, "score", 1.0),
                        reading_order=getattr(raw_el, "reading_order", order_idx),
                    )
                elements.append(el)

            markdown = assemble_markdown(elements)
            pages.append(PageResult(page_num=page_num, elements=elements, markdown=markdown))

        total_elements = sum(len(p.elements) for p in pages)
        return cls(source_file=source_file, pages=pages, total_elements=total_elements)

    def save(self, output_dir: Path) -> None:
        """Save this result as Markdown and JSON files.

        Args:
            output_dir: Directory to write output files.
        """
        save_to_json(self, output_dir)


class DocumentParser:
    """Main document parsing pipeline using GLM-OCR MaaS API.

    Wraps the glmocr SDK to provide structured document parsing
    via the Z.AI cloud API (PP-DocLayout-V3 + GLM-OCR 0.9B).

    Example:
        parser = DocumentParser()
        result = parser.parse_file(Path("document.pdf"))
        result.save(Path("./output"))
    """

    def __init__(self) -> None:
        """Initialize the DocumentParser with settings from environment."""
        if not _GLMOCR_AVAILABLE:
            raise ImportError(
                "glmocr package is required. Install with: uv pip install glmocr"
            )
        settings = get_settings()
        self._parser = GlmOcr(
            config_path=settings.config_yaml_path,
            api_key=settings.z_ai_api_key.get_secret_value(),
        )
        logger.info("DocumentParser initialized with config: %s", settings.config_yaml_path)

    def parse_file(self, file_path: str | Path) -> ParseResult:
        """Parse a single PDF or image file.

        Args:
            file_path: Path to the document to parse (PDF or image).

        Returns:
            ParseResult with structured elements and assembled Markdown.

        Raises:
            FileNotFoundError: If the file does not exist.
            ImportError: If glmocr is not installed.
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        logger.info("Parsing file: %s", file_path)
        raw = self._parser.parse(str(file_path))
        result = ParseResult.from_sdk_result(raw, source_file=str(file_path))
        logger.info(
            "Parsed %s: %d pages, %d elements",
            file_path.name,
            len(result.pages),
            result.total_elements,
        )
        return result

    def parse_batch(
        self,
        file_paths: list[Path],
        output_dir: Path,
    ) -> list[ParseResult]:
        """Parse multiple files with progress tracking.

        Args:
            file_paths: List of paths to documents to parse.
            output_dir: Directory to save output files.

        Returns:
            List of ParseResult objects, one per input file.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        results: list[ParseResult] = []

        for fp in tqdm(file_paths, desc="Parsing documents", unit="file"):
            try:
                result = self.parse_file(fp)
                result.save(output_dir)
                results.append(result)
            except Exception as e:
                logger.error("Failed to parse %s: %s", fp, e, exc_info=True)
                raise

        return results
