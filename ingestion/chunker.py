"""Deterministic infobox-first document chunking.

Chunks are a text substrate. They do not create Event records. Olympic-looking
titles without an infobox remain document-only and still receive chunks.
"""

from __future__ import annotations

import re

from ingestion.models import ChunkingConfig, ParsedDocument, TextChunk
from ingestion.parse import INFOBOX_HEADER_RE, INFOBOX_LINE_RE

HEADING_RE = re.compile(
    r"^(?:"
    r"Results(?: table)?|Schedule|Records|Qualification|Summary|Background|"
    r"Course|Medalists|Medallists|Overview|Format|Competition format|"
    r"Heats|Heat \d+|Semifinals?|Semifinal \d+|Quarterfinals?|Quarterfinal \d+|"
    r"Finals?|Final [A-D]|Round \d+|See also|References|External links|Notes|"
    r"Plot|Cast|Production|Reception|History|Legacy|Controversy"
    r")$",
    re.IGNORECASE,
)
TITLE_CASE_HEADING_RE = re.compile(r"^[A-Z0-9][A-Za-z0-9 ,/&'()+.\-]{1,68}$")
TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def chunk_document(document: ParsedDocument, config: ChunkingConfig | None = None) -> list[TextChunk]:
    cfg = config or ChunkingConfig()
    text = document.text or ""
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    pieces: list[tuple[str, str, int, int]] = []
    infobox_text, infobox_span, body = _split_infobox(normalized)
    if infobox_text and infobox_span is not None:
        pieces.append(("infobox", infobox_text, infobox_span[0], infobox_span[1]))
    pieces.extend(_section_pieces(body, offset=_body_offset(normalized, body)))
    if not pieces and normalized.strip():
        pieces.append(("body", normalized.strip(), 0, len(normalized)))
    windows = _window_pieces(pieces, cfg)
    windows = _merge_tiny(windows, cfg)
    event_id = document.event.event_id if document.event is not None else None
    chunks: list[TextChunk] = []
    for index, (section, kind, span_text, start, end) in enumerate(windows):
        chunk_id = f"{document.doc_id}::c{index:03d}"
        indexed = span_text
        if cfg.include_title_in_index and document.title:
            indexed = f"{document.title}\n{span_text}"
        chunks.append(
            TextChunk(
                chunk_id=chunk_id,
                document_id=document.doc_id,
                document_title=document.title,
                source_url=document.url,
                text=span_text,
                indexed_text=indexed,
                start_char=start,
                end_char=end,
                token_count=len(TOKEN_RE.findall(span_text)),
                chunk_index=index,
                section=section,
                kind=kind,
                event_id=event_id,
                document_kind=document.kind,
            )
        )
    return chunks


def chunk_corpus(documents: list[ParsedDocument], config: ChunkingConfig | None = None) -> list[TextChunk]:
    chunks: list[TextChunk] = []
    for document in documents:
        chunks.extend(chunk_document(document, config))
    return chunks


def chunk_stats(chunks: list[TextChunk]) -> dict:
    if not chunks:
        return {
            "total_chunks": 0,
            "avg_chars": 0.0,
            "median_chars": 0,
            "min_chars": 0,
            "max_chars": 0,
            "avg_tokens": 0.0,
        }
    lengths = sorted(len(chunk.text) for chunk in chunks)
    tokens = [chunk.token_count for chunk in chunks]
    mid = len(lengths) // 2
    median = lengths[mid] if len(lengths) % 2 else (lengths[mid - 1] + lengths[mid]) / 2
    return {
        "total_chunks": len(chunks),
        "avg_chars": round(sum(lengths) / len(lengths), 1),
        "median_chars": median,
        "min_chars": lengths[0],
        "max_chars": lengths[-1],
        "avg_tokens": round(sum(tokens) / len(tokens), 1),
    }


def _split_infobox(text: str) -> tuple[str | None, tuple[int, int] | None, str]:
    header = INFOBOX_HEADER_RE.search(text)
    if not header:
        return None, None, text
    start = header.start()
    cursor = header.end()
    if cursor < len(text) and text[cursor] == "\n":
        cursor += 1
    end = cursor
    for line in text[cursor:].split("\n"):
        if not line.strip():
            break
        if not INFOBOX_LINE_RE.match(line):
            break
        end += len(line) + 1
    infobox = text[start:end].strip()
    body = text[end:].lstrip("\n")
    return infobox or None, (start, start + len(infobox)) if infobox else None, body


def _body_offset(full: str, body: str) -> int:
    if not body:
        return len(full)
    found = full.find(body)
    return found if found >= 0 else 0


def _section_pieces(body: str, offset: int) -> list[tuple[str, str, int, int]]:
    if not body.strip():
        return []
    lines = body.split("\n")
    sections: list[list[str]] = []
    names: list[str] = []
    current: list[str] = []
    current_name = "lead"
    for line in lines:
        if _is_heading(line):
            if any(part.strip() for part in current):
                sections.append(current)
                names.append(current_name)
            current = [line]
            current_name = line.strip()
            continue
        current.append(line)
    if any(part.strip() for part in current):
        sections.append(current)
        names.append(current_name)
    pieces: list[tuple[str, str, int, int]] = []
    cursor = offset
    for name, block_lines in zip(names, sections):
        raw = "\n".join(block_lines).strip("\n")
        # Map back onto the original body by searching from cursor.
        rel = body.find(raw, max(0, cursor - offset))
        start = offset + rel if rel >= 0 else cursor
        end = start + len(raw)
        kind = "lead" if name == "lead" else "section"
        pieces.append((name, raw.strip(), start, end))
        cursor = end
    return [(name, text, start, end) for name, text, start, end in pieces if text.strip()]


def _is_heading(line: str) -> bool:
    text = line.strip()
    if not text or "|" in text or text.startswith("["):
        return False
    if text[-1] in ".;,:":
        return False
    words = text.split()
    if len(words) > 7 or len(text) > 70:
        return False
    if HEADING_RE.match(text):
        return True
    if not TITLE_CASE_HEADING_RE.match(text):
        return False
    lowered = {word.casefold() for word in words}
    if lowered & {"the", "a", "an", "is", "are", "was", "were", "of"} and len(words) > 3:
        return False
    return all(word[:1].isupper() or word[:1].isdigit() for word in words if word not in {"and", "of", "at", "the"})


def _window_pieces(
    pieces: list[tuple[str, str, int, int]],
    cfg: ChunkingConfig,
) -> list[tuple[str, str, str, int, int]]:
    windows: list[tuple[str, str, str, int, int]] = []
    for section, text, start, end in pieces:
        kind = "infobox" if section == "infobox" else ("lead" if section == "lead" else "section")
        if section == "infobox" or len(text) <= cfg.max_chars:
            windows.append((section, kind, text, start, end))
            continue
        paragraphs = [part for part in re.split(r"\n\s*\n", text) if part.strip()]
        if len(paragraphs) <= 1 and len(text) <= cfg.hard_max_chars:
            windows.append((section, kind, text, start, end))
            continue
        buf = ""
        buf_start = start
        consumed = start
        for para in paragraphs:
            candidate = para if not buf else f"{buf}\n\n{para}"
            if len(candidate) <= cfg.max_chars or not buf:
                if not buf:
                    buf_start = consumed
                buf = candidate
            else:
                windows.append((section, kind, buf, buf_start, buf_start + len(buf)))
                buf = para
                buf_start = consumed
            consumed += len(para)
        if buf.strip():
            if len(buf) > cfg.hard_max_chars:
                windows.extend(_hard_split(section, kind, buf, buf_start, cfg.hard_max_chars))
            else:
                windows.append((section, kind, buf, buf_start, buf_start + len(buf)))
    return windows


def _hard_split(
    section: str,
    kind: str,
    text: str,
    start: int,
    limit: int,
) -> list[tuple[str, str, str, int, int]]:
    parts: list[tuple[str, str, str, int, int]] = []
    cursor = 0
    while cursor < len(text):
        end = min(len(text), cursor + limit)
        if end < len(text):
            space = text.rfind(" ", cursor + limit // 2, end)
            if space > cursor:
                end = space
        chunk = text[cursor:end].strip()
        if chunk:
            parts.append((section, kind, chunk, start + cursor, start + end))
        cursor = end
        while cursor < len(text) and text[cursor].isspace():
            cursor += 1
    return parts


def _merge_tiny(
    windows: list[tuple[str, str, str, int, int]],
    cfg: ChunkingConfig,
) -> list[tuple[str, str, str, int, int]]:
    if not windows:
        return []
    merged: list[tuple[str, str, str, int, int]] = []
    for section, kind, text, start, end in windows:
        if (
            merged
            and kind != "infobox"
            and merged[-1][1] != "infobox"
            and kind == merged[-1][1]
            and section == merged[-1][0]
            and len(text) < cfg.min_chars
            and len(merged[-1][2]) + len(text) + 2 <= cfg.hard_max_chars
        ):
            prev_section, prev_kind, prev_text, prev_start, _prev_end = merged[-1]
            merged[-1] = (prev_section, prev_kind, f"{prev_text}\n\n{text}", prev_start, end)
            continue
        merged.append((section, kind, text, start, end))
    collapsed: list[tuple[str, str, str, int, int]] = []
    for section, kind, text, start, end in merged:
        if (
            collapsed
            and kind != "infobox"
            and len(collapsed[-1][2]) < cfg.min_chars
            and collapsed[-1][1] != "infobox"
            and len(collapsed[-1][2]) + len(text) + 2 <= cfg.hard_max_chars
        ):
            prev_section, prev_kind, prev_text, prev_start, _prev_end = collapsed[-1]
            collapsed[-1] = (prev_section, prev_kind, f"{prev_text}\n\n{text}", prev_start, end)
            continue
        collapsed.append((section, kind, text, start, end))
    return collapsed
