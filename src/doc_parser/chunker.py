"""Structure-aware chunker for RAG-ready document chunks."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Protocol

logger = logging.getLogger(__name__)

# Labels that must never be split across chunks
ATOMIC_LABELS: frozenset[str] = frozenset({"table", "formula", "inline_formula", "algorithm"})

# Labels that are headings/titles — attach to following content
TITLE_LABELS: frozenset[str] = frozenset({"document_title", "paragraph_title"})


@dataclass
class Chunk:
    """A RAG-ready document chunk.

    Attributes:
        text: The chunk text content.
        chunk_id: Unique identifier in format "{source_file}_{page}_{idx}".
        page: Page number the chunk came from.
        element_types: List of element labels included in this chunk.
        bbox: Bounding box [x1, y1, x2, y2] or None if multi-element chunk.
        source_file: Source document filename.
        is_atomic: True for atomic elements (tables, formulas) that must not be split.
    """

    text: str
    chunk_id: str
    page: int
    element_types: list[str]
    bbox: list[float] | None
    source_file: str
    is_atomic: bool


class ElementLike(Protocol):
    """Duck-typing protocol for parsed elements."""

    label: str
    text: str
    bbox: list[float]
    score: float
    reading_order: int


def _estimate_tokens(text: str) -> int:
    """Estimate token count using word count * 1.3 heuristic."""
    return int(len(text.split()) * 1.3)


def _split_text_into_sub_chunks(text: str, max_tokens: int) -> list[str]:
    """Split a single large text block into sub-chunks that fit within max_tokens.

    Splits on whitespace boundaries to avoid cutting mid-word.

    Args:
        text: The text to split.
        max_tokens: Maximum tokens per sub-chunk.

    Returns:
        List of text sub-chunks each within the token limit.
    """
    words = text.split()
    # Approximate words per chunk: max_tokens / 1.3
    words_per_chunk = max(1, int(max_tokens / 1.3))

    sub_chunks = []
    for i in range(0, len(words), words_per_chunk):
        sub_chunks.append(" ".join(words[i : i + words_per_chunk]))
    return sub_chunks


def structure_aware_chunking(
    elements: list[Any],
    source_file: str,
    page: int,
    max_chunk_tokens: int = 512,
) -> list[Chunk]:
    """Chunk document elements respecting structure boundaries.

    Rules:
    - Atomic elements (table, formula, algorithm) → always their own chunk
    - Title elements attach to the following content element (no orphan headings)
    - Text elements are accumulated until max_chunk_tokens is reached
    - When a chunk overflows, start a new chunk

    Args:
        elements: List of parsed elements with label, text, bbox, reading_order.
        source_file: Source document filename for chunk_id generation.
        page: Page number for chunk_id generation.
        max_chunk_tokens: Maximum tokens per chunk (default 512).

    Returns:
        List of Chunk objects ready for vector store ingestion.
    """
    if not elements:
        return []

    sorted_elements = sorted(elements, key=lambda e: e.reading_order)
    chunks: list[Chunk] = []
    chunk_idx = 0

    # Accumulator for current chunk
    current_texts: list[str] = []
    current_labels: list[str] = []
    current_tokens: int = 0
    pending_title: str | None = None
    pending_title_label: str | None = None

    def flush_current(force: bool = False) -> None:
        """Flush accumulated text into a Chunk if non-empty."""
        nonlocal current_texts, current_labels, current_tokens, chunk_idx
        nonlocal pending_title, pending_title_label

        if not current_texts and pending_title is None:
            return

        # Include pending title if it never got attached
        texts_to_flush = []
        labels_to_flush = []
        if pending_title is not None:
            texts_to_flush.append(pending_title)
            labels_to_flush.append(pending_title_label or "paragraph_title")
            pending_title = None
            pending_title_label = None

        texts_to_flush.extend(current_texts)
        labels_to_flush.extend(current_labels)

        if not texts_to_flush:
            return

        chunk_text = "\n\n".join(texts_to_flush)
        chunk = Chunk(
            text=chunk_text,
            chunk_id=f"{source_file}_{page}_{chunk_idx}",
            page=page,
            element_types=labels_to_flush,
            bbox=None,  # multi-element chunks don't have a single bbox
            source_file=source_file,
            is_atomic=False,
        )
        chunks.append(chunk)
        chunk_idx += 1
        current_texts = []
        current_labels = []
        current_tokens = 0

    for element in sorted_elements:
        label = element.label
        text = element.text.strip()

        if not text:
            continue

        # Atomic elements → flush current, emit atomic chunk, continue
        if label in ATOMIC_LABELS:
            flush_current()
            atomic_chunk = Chunk(
                text=text,
                chunk_id=f"{source_file}_{page}_{chunk_idx}",
                page=page,
                element_types=[label],
                bbox=element.bbox,
                source_file=source_file,
                is_atomic=True,
            )
            chunks.append(atomic_chunk)
            chunk_idx += 1
            continue

        # Title elements → store as pending; will attach to next content
        if label in TITLE_LABELS:
            # If there's already pending content + a title, flush first
            if current_texts:
                flush_current()
            # If there's already a pending title with no content, flush it
            elif pending_title is not None:
                flush_current()
            pending_title = text
            pending_title_label = label
            continue

        # Regular content element
        token_estimate = _estimate_tokens(text)
        # Add pending title tokens if present
        pending_tokens = _estimate_tokens(pending_title) if pending_title else 0

        # If a single element alone exceeds the limit, split it into sub-chunks
        if token_estimate > max_chunk_tokens:
            # Flush any accumulated content first
            flush_current()
            sub_chunks = _split_text_into_sub_chunks(text, max_chunk_tokens)
            for sub_text in sub_chunks:
                chunk = Chunk(
                    text=sub_text,
                    chunk_id=f"{source_file}_{page}_{chunk_idx}",
                    page=page,
                    element_types=[label],
                    bbox=None,
                    source_file=source_file,
                    is_atomic=False,
                )
                chunks.append(chunk)
                chunk_idx += 1
            continue

        # Check if adding this element would overflow
        if current_texts and (current_tokens + token_estimate + pending_tokens > max_chunk_tokens):
            flush_current()

        # Attach pending title to this content element
        if pending_title is not None:
            current_texts.append(pending_title)
            current_labels.append(pending_title_label or "paragraph_title")
            current_tokens += _estimate_tokens(pending_title)
            pending_title = None
            pending_title_label = None

        current_texts.append(text)
        current_labels.append(label)
        current_tokens += token_estimate

        # If we're now over the limit, flush
        if current_tokens >= max_chunk_tokens:
            flush_current()

    # Flush any remaining content
    flush_current()

    return chunks
