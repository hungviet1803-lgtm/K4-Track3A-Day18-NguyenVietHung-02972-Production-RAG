from __future__ import annotations

"""
Module 1: Advanced Chunking Strategies
=======================================
Implement semantic, hierarchical, và structure-aware chunking.
So sánh với basic chunking (baseline) để thấy improvement.

Test: pytest tests/test_m1.py
"""

import os, sys, glob, re
from dataclasses import dataclass, field

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (DATA_DIR, HIERARCHICAL_PARENT_SIZE, HIERARCHICAL_CHILD_SIZE,
                    SEMANTIC_THRESHOLD)


@dataclass
class Chunk:
    text: str
    metadata: dict = field(default_factory=dict)
    parent_id: str | None = None


def _extract_pdf_text(path: str) -> str:
    """Extract text layer từ PDF. Trả về "" nếu PDF là scan ảnh (không có text)."""
    from pypdf import PdfReader

    reader = PdfReader(path)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages).strip()


def load_documents(data_dir: str = DATA_DIR) -> list[dict]:
    """Load tất cả markdown và PDF (có text layer) từ data/. (Đã implement sẵn)

    - .md: đọc trực tiếp.
    - .pdf: trích text layer bằng pypdf. PDF scan ảnh (không có text) bị bỏ qua
      kèm cảnh báo — RAG text-based không xử lý được scan nếu chưa OCR.
    """
    docs = []
    for fp in sorted(glob.glob(os.path.join(data_dir, "*.md"))):
        with open(fp, encoding="utf-8") as f:
            docs.append({"text": f.read(), "metadata": {"source": os.path.basename(fp)}})

    for fp in sorted(glob.glob(os.path.join(data_dir, "*.pdf"))):
        text = _extract_pdf_text(fp)
        if text:
            docs.append({"text": text, "metadata": {"source": os.path.basename(fp)}})
        else:
            print(f"  ⚠️  Bỏ qua {os.path.basename(fp)}: PDF scan ảnh, không có text layer (cần OCR).")

    return docs


# ─── Baseline: Basic Chunking (để so sánh) ──────────────


def chunk_basic(text: str, chunk_size: int = 500, metadata: dict | None = None) -> list[Chunk]:
    """
    Basic chunking: split theo paragraph (\\n\\n).
    Đây là baseline — KHÔNG phải mục tiêu của module này.
    (Đã implement sẵn)
    """
    metadata = metadata or {}
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current = ""
    for i, para in enumerate(paragraphs):
        if len(current) + len(para) > chunk_size and current:
            chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
            current = ""
        current += para + "\n\n"
    if current.strip():
        chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
    return chunks


# ─── Strategy 1: Semantic Chunking ───────────────────────


SENTENCE_SPLIT_RE = re.compile(r'(?<=[.!?])\s+|\n\n')
SEMANTIC_MODEL_NAME = "all-MiniLM-L6-v2"
_semantic_model = None


def _get_semantic_model():
    """Load model một lần rồi cache — encode lại model mỗi lần gọi rất chậm."""
    global _semantic_model
    if _semantic_model is None:
        from sentence_transformers import SentenceTransformer
        _semantic_model = SentenceTransformer(SEMANTIC_MODEL_NAME)
    return _semantic_model


def _split_sentences(text: str) -> list[tuple[int, int]]:
    """Tách câu, trả về (start, end) offset trong text gốc (bỏ câu rỗng).

    Giữ offset thay vì chuỗi để khi gộp nhóm có thể cắt lại đúng đoạn text gốc —
    không làm mất xuống dòng, bảng, danh sách như khi " ".join(sentences).
    """
    spans, start = [], 0
    for m in SENTENCE_SPLIT_RE.finditer(text):
        spans.append((start, m.start()))
        start = m.end()
    spans.append((start, len(text)))

    result = []
    for s, e in spans:
        seg = text[s:e]
        if not seg.strip():
            continue
        # Thu hẹp span về phần không phải khoảng trắng
        s += len(seg) - len(seg.lstrip())
        e -= len(seg) - len(seg.rstrip())
        result.append((s, e))
    return result


def chunk_semantic(text: str, threshold: float = SEMANTIC_THRESHOLD,
                   metadata: dict | None = None) -> list[Chunk]:
    """
    Split text by sentence similarity — nhóm câu cùng chủ đề.
    Tốt hơn basic vì không cắt giữa ý.

    Câu i được gộp vào nhóm hiện tại nếu cosine(câu i-1, câu i) >= threshold,
    ngược lại coi là chuyển ý → mở chunk mới.
    """
    from numpy import dot
    from numpy.linalg import norm

    metadata = metadata or {}
    spans = _split_sentences(text)
    if not spans:
        return []

    sentences = [text[s:e] for s, e in spans]
    embeddings = _get_semantic_model().encode(sentences, batch_size=64, show_progress_bar=False)

    def cosine_sim(a, b) -> float:
        return float(dot(a, b) / (norm(a) * norm(b) + 1e-9))

    groups = [[0]]
    for i in range(1, len(sentences)):
        if cosine_sim(embeddings[i - 1], embeddings[i]) < threshold:
            groups.append([i])
        else:
            groups[-1].append(i)

    chunks = []
    for group in groups:
        start, end = spans[group[0]][0], spans[group[-1]][1]
        chunks.append(Chunk(
            text=text[start:end],
            metadata={**metadata, "strategy": "semantic", "chunk_index": len(chunks),
                      "sentence_count": len(group)},
        ))
    return chunks


# ─── Strategy 2: Hierarchical Chunking ──────────────────


def chunk_hierarchical(text: str, parent_size: int = HIERARCHICAL_PARENT_SIZE,
                       child_size: int = HIERARCHICAL_CHILD_SIZE,
                       metadata: dict | None = None) -> tuple[list[Chunk], list[Chunk]]:
    """
    Parent-child hierarchy: retrieve child (precision) → return parent (context).
    Đây là default recommendation cho production RAG.

    Returns:
        (parents, children) — mỗi child có parent_id link đến parent.
    """
    metadata = metadata or {}
    parents, children = [], []

    for parent_text in _split_to_size(text, parent_size):
        pid = f"parent_{len(parents)}"
        parents.append(Chunk(
            text=parent_text,
            metadata={**metadata, "strategy": "hierarchical", "chunk_type": "parent",
                      "parent_id": pid, "chunk_index": len(parents)},
        ))
        for child_text in _split_to_size(parent_text, child_size):
            children.append(Chunk(
                text=child_text,
                metadata={**metadata, "strategy": "hierarchical", "chunk_type": "child",
                          "parent_id": pid, "chunk_index": len(children)},
                parent_id=pid,
            ))

    return parents, children


# Thứ tự ưu tiên điểm cắt: đoạn → dòng (giữ nguyên hàng của bảng/list) → câu → từ.
_SPLIT_LEVELS = [
    (re.compile(r'\n\n+'), "\n\n"),
    (re.compile(r'\n'), "\n"),
    (re.compile(r'(?<=[.!?])\s+'), " "),
    (re.compile(r' +'), " "),
]


def _split_to_size(text: str, max_size: int, level: int = 0) -> list[str]:
    """Chia text thành các đoạn ≤ max_size, cắt ở ranh giới tự nhiên lớn nhất có thể.

    Gộp tham lam các mảnh ở cùng cấp; mảnh nào tự nó vượt max_size thì đệ quy
    xuống cấp tách nhỏ hơn. Chỉ cắt cứng theo ký tự khi một "từ" dài hơn max_size.
    """
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_size:
        return [text]
    if level >= len(_SPLIT_LEVELS):
        return [text[i:i + max_size] for i in range(0, len(text), max_size)]

    pattern, joiner = _SPLIT_LEVELS[level]
    pieces = [p.strip() for p in pattern.split(text) if p.strip()]

    out, current = [], ""
    for piece in pieces:
        if len(piece) > max_size:
            if current:
                out.append(current)
                current = ""
            out.extend(_split_to_size(piece, max_size, level + 1))
        elif not current:
            current = piece
        elif len(current) + len(joiner) + len(piece) <= max_size:
            current += joiner + piece
        else:
            out.append(current)
            current = piece
    if current:
        out.append(current)
    return out


# ─── Strategy 3: Structure-Aware Chunking ────────────────


HEADER_RE = re.compile(r'^(#{1,3})\s+(.+?)\s*#*\s*$')


def chunk_structure_aware(text: str, metadata: dict | None = None) -> list[Chunk]:
    """
    Parse markdown headers → chunk theo logical structure.
    Giữ nguyên tables, code blocks, lists — không cắt giữa chừng.
    """
    metadata = metadata or {}
    chunks: list[Chunk] = []
    path: list[tuple[int, str]] = []   # stack (level, title) của các header đang mở
    header_line: str | None = None
    body: list[str] = []
    in_fence = False

    def flush():
        content = "\n".join(body).strip()
        # Header không có nội dung riêng (vd "# Nghỉ phép" ngay trước "## ...")
        # không thành chunk; tên của nó vẫn nằm trong section_path của các mục con.
        if not content:
            return
        title = path[-1][1] if path else ""
        chunks.append(Chunk(
            text=f"{header_line}\n{content}" if header_line else content,
            metadata={**metadata, "strategy": "structure", "chunk_index": len(chunks),
                      "section": title,
                      "section_path": " > ".join(t for _, t in path),
                      "header_level": path[-1][0] if path else 0},
        ))

    for line in text.splitlines():
        # Dòng "# ..." bên trong code block không phải header
        if line.lstrip().startswith(("```", "~~~")):
            in_fence = not in_fence
        m = None if in_fence else HEADER_RE.match(line)
        if m:
            flush()
            level, title = len(m.group(1)), m.group(2).strip()
            while path and path[-1][0] >= level:
                path.pop()
            path.append((level, title))
            header_line, body = line.strip(), []
        else:
            body.append(line)
    flush()
    return chunks


# ─── A/B Test: Compare All Strategies ────────────────────


def compare_strategies(documents: list[dict]) -> dict:
    """
    Run all strategies on documents and compare.
    (Đã implement sẵn — sẽ hoạt động khi bạn implement 3 strategies ở trên)
    """
    def _stats(chunk_list):
        lengths = [len(c.text) for c in chunk_list]
        if not lengths:
            return {"count": 0, "avg_len": 0, "min_len": 0, "max_len": 0}
        return {
            "count": len(lengths),
            "avg_len": round(sum(lengths) / len(lengths)),
            "min_len": min(lengths),
            "max_len": max(lengths),
        }

    all_text = "\n\n".join(d["text"] for d in documents)
    meta = {"source": "all"}

    basic = chunk_basic(all_text, metadata=meta)
    semantic = chunk_semantic(all_text, metadata=meta)
    parents, children = chunk_hierarchical(all_text, metadata=meta)
    structure = chunk_structure_aware(all_text, metadata=meta)

    results = {
        "basic": _stats(basic),
        "semantic": _stats(semantic),
        "hierarchical": {**_stats(children), "parents": len(parents)},
        "structure": _stats(structure),
    }

    print(f"{'Strategy':<15} {'Chunks':>7} {'Avg':>5} {'Min':>5} {'Max':>5}")
    for name, s in results.items():
        print(f"{name:<15} {s['count']:>7} {s['avg_len']:>5} {s['min_len']:>5} {s['max_len']:>5}")

    return results


if __name__ == "__main__":
    docs = load_documents()
    print(f"Loaded {len(docs)} documents")
    results = compare_strategies(docs)
    for name, stats in results.items():
        print(f"  {name}: {stats}")
