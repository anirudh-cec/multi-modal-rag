"""Unit tests for chunker.structure_aware_chunking()."""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


@dataclass
class ParsedElement:
    label: str
    text: str
    bbox: list[float]
    score: float
    reading_order: int


class TestStructureAwareChunking:
    def test_table_produces_single_atomic_chunk(self):
        """A table element becomes exactly 1 chunk with is_atomic=True."""
        from doc_parser.chunker import structure_aware_chunking
        table_html = "<table><tr><td>A</td><td>B</td></tr></table>"
        elements = [ParsedElement("table", table_html, [0, 0, 1, 1], 0.9, 0)]
        chunks = structure_aware_chunking(elements, source_file="test.pdf", page=1)
        assert len(chunks) == 1
        assert chunks[0].is_atomic is True

    def test_formula_produces_single_atomic_chunk(self):
        """A formula element becomes exactly 1 chunk with is_atomic=True."""
        from doc_parser.chunker import structure_aware_chunking
        elements = [ParsedElement("formula", "E = mc^2", [0,0,1,1], 0.9, 0)]
        chunks = structure_aware_chunking(elements, source_file="test.pdf", page=1)
        assert len(chunks) == 1
        assert chunks[0].is_atomic is True

    def test_algorithm_produces_single_atomic_chunk(self):
        """An algorithm element becomes exactly 1 chunk with is_atomic=True."""
        from doc_parser.chunker import structure_aware_chunking
        elements = [ParsedElement("algorithm", "procedure Sort(A)", [0,0,1,1], 0.9, 0)]
        chunks = structure_aware_chunking(elements, source_file="test.pdf", page=1)
        assert len(chunks) == 1
        assert chunks[0].is_atomic is True

    def test_title_attaches_to_following_paragraph(self):
        """A paragraph_title is NOT an orphan chunk — it attaches to the next content."""
        from doc_parser.chunker import structure_aware_chunking
        elements = [
            ParsedElement("paragraph_title", "Introduction", [0,0,1,1], 0.9, 0),
            ParsedElement("paragraph", "This section covers...", [0,0,1,1], 0.9, 1),
        ]
        chunks = structure_aware_chunking(elements, source_file="test.pdf", page=1)
        # Title + paragraph should be in the same chunk
        assert len(chunks) == 1
        assert "Introduction" in chunks[0].text
        assert "This section covers..." in chunks[0].text

    def test_max_chunk_tokens_boundary(self):
        """Text longer than max_chunk_tokens gets split into multiple chunks."""
        from doc_parser.chunker import structure_aware_chunking
        # ~600 words → should exceed default 512 token limit
        long_text = " ".join(["word"] * 600)
        elements = [ParsedElement("paragraph", long_text, [0,0,1,1], 0.9, 0)]
        chunks = structure_aware_chunking(
            elements, source_file="test.pdf", page=1, max_chunk_tokens=512
        )
        assert len(chunks) >= 2

    def test_chunk_id_format(self):
        """chunk_id must follow '{source_file}_{page}_{idx}' pattern."""
        from doc_parser.chunker import structure_aware_chunking
        elements = [ParsedElement("paragraph", "Hello", [0,0,1,1], 0.9, 0)]
        chunks = structure_aware_chunking(elements, source_file="myfile.pdf", page=3)
        assert chunks[0].chunk_id.startswith("myfile.pdf_3_")

    def test_chunk_has_correct_metadata(self):
        """Chunk has correct page, source_file, and element_types."""
        from doc_parser.chunker import structure_aware_chunking
        elements = [ParsedElement("paragraph", "Test text", [0,0,1,1], 0.9, 0)]
        chunks = structure_aware_chunking(elements, source_file="doc.pdf", page=2)
        assert chunks[0].page == 2
        assert chunks[0].source_file == "doc.pdf"
        assert "paragraph" in chunks[0].element_types

    def test_non_atomic_chunk_is_not_atomic(self):
        """Regular paragraphs have is_atomic=False."""
        from doc_parser.chunker import structure_aware_chunking
        elements = [ParsedElement("paragraph", "Regular text", [0,0,1,1], 0.9, 0)]
        chunks = structure_aware_chunking(elements, source_file="test.pdf", page=1)
        assert chunks[0].is_atomic is False

    def test_empty_elements_returns_empty_list(self):
        """Empty input produces empty output."""
        from doc_parser.chunker import structure_aware_chunking
        chunks = structure_aware_chunking([], source_file="test.pdf", page=1)
        assert chunks == []
