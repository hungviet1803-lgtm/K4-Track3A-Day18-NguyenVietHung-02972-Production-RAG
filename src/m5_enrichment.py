from __future__ import annotations

"""
Module 5: Enrichment Pipeline
==============================
Làm giàu chunks TRƯỚC khi embed: Summarize, HyQA, Contextual Prepend, Auto Metadata.

Test: pytest tests/test_m5.py
"""

import os, sys, json
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import LLM_API_KEY, LLM_MODEL, make_llm_client


@dataclass
class EnrichedChunk:
    """Chunk đã được làm giàu."""
    original_text: str
    enriched_text: str
    summary: str
    hypothesis_questions: list[str]
    auto_metadata: dict
    method: str  # "contextual", "summary", "hyqa", "full"


# ─── Shared helpers ──────────────────────────────────────

_client = None


def _get_client():
    """OpenAI client dùng chung (tạo 1 lần cho cả trăm chunk)."""
    global _client
    if _client is None:
        _client = make_llm_client()
    return _client


def _chat(system: str, user: str, max_tokens: int, json_mode: bool = False) -> str:
    kwargs = {"response_format": {"type": "json_object"}} if json_mode else {}
    resp = _get_client().chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        max_tokens=max_tokens,
        temperature=0,
        **kwargs,
    )
    return (resp.choices[0].message.content or "").strip()


def _loads_lenient(content: str):
    """json.loads chịu lỗi: Gemini (json mode) đôi khi để dấu phẩy thừa trước } hoặc ]."""
    import re
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return json.loads(re.sub(r",\s*([}\]])", r"\1", content))


def _sentences(text: str) -> list[str]:
    """Câu "thật" của chunk: bỏ header markdown, dòng bảng, dòng metadata (> ...)."""
    import re
    lines = [ln.strip() for ln in text.splitlines()]
    prose = " ".join(ln for ln in lines if ln and not ln.startswith(("#", "|", ">")))
    prose = prose.replace("**", "")
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", prose) if len(s.strip()) > 10]


# ─── Technique 1: Chunk Summarization ────────────────────


def summarize_chunk(text: str) -> str:
    """
    Tạo summary ngắn cho chunk.
    Embed summary thay vì (hoặc cùng với) raw chunk → giảm noise.
    """
    if LLM_API_KEY:
        try:
            return _chat("Tóm tắt đoạn văn sau trong 2-3 câu ngắn gọn bằng tiếng Việt. "
                         "Giữ nguyên mọi con số, mốc thời gian và điều kiện.", text, max_tokens=150)
        except Exception as e:
            print(f"  ⚠️  OpenAI summarize failed: {e}")

    return _extractive_summary(text)


def _extractive_summary(text: str) -> str:
    """Fallback: 2 câu đầu (không tính header/bảng)."""
    sentences = _sentences(text)
    return " ".join(sentences[:2]) if sentences else text.strip()


# ─── Technique 2: Hypothesis Question-Answer (HyQA) ─────


def generate_hypothesis_questions(text: str, n_questions: int = 3) -> list[str]:
    """
    Generate câu hỏi mà chunk có thể trả lời.
    Index cả questions lẫn chunk → query match tốt hơn (bridge vocabulary gap).
    """
    if LLM_API_KEY:
        try:
            content = _chat(f"Dựa trên đoạn văn, tạo {n_questions} câu hỏi tiếng Việt mà nhân viên "
                            "có thể hỏi và đoạn văn trả lời được. Mỗi câu hỏi trên 1 dòng, không đánh số.",
                            text, max_tokens=200)
            questions = [q.strip().lstrip("0123456789.-) ").strip() for q in content.split("\n")]
            return [q for q in questions if q][:n_questions]
        except Exception as e:
            print(f"  ⚠️  OpenAI HyQA failed: {e}")

    # Fallback: biến câu khẳng định thành dạng hỏi — kém tự nhiên hơn LLM nhưng vẫn
    # thêm được từ khóa của chunk dưới dạng câu hỏi.
    return [f"{s.rstrip('.!?')}?" for s in _sentences(text)[:n_questions]]


# ─── Technique 3: Contextual Prepend (Anthropic style) ──


def contextual_prepend(text: str, document_title: str = "") -> str:
    """
    Prepend context giải thích chunk nằm ở đâu trong document.
    Anthropic benchmark: giảm 49% retrieval failure (alone).
    """
    if LLM_API_KEY:
        try:
            context = _chat("Viết 1 câu ngắn mô tả đoạn văn này nằm ở đâu trong tài liệu và nói về "
                            "chủ đề gì. Chỉ trả về 1 câu.",
                            f"Tài liệu: {document_title}\n\nĐoạn văn:\n{text}", max_tokens=80)
            if context:
                return f"{context}\n\n{text}"
        except Exception as e:
            print(f"  ⚠️  OpenAI contextual failed: {e}")

    # Fallback: ít nhất gắn tên tài liệu để chunk không "mồ côi" chủ đề
    prefix = f"Trích từ {document_title}.\n\n" if document_title else ""
    return f"{prefix}{text}"


# ─── Technique 4: Auto Metadata Extraction ──────────────

_CATEGORY_KEYWORDS = {
    "it": ["mật khẩu", "vpn", "bảo mật", "dữ liệu", "mfa", "hệ thống", "cntt", "truy cập"],
    "finance": ["lương", "chi phí", "tạm ứng", "thưởng", "phụ cấp", "vnđ", "hoàn chi", "mua sắm"],
    "hr": ["nghỉ phép", "thử việc", "đánh giá", "đào tạo", "mentor", "nhân viên mới", "bảo hiểm"],
}


def extract_metadata(text: str) -> dict:
    """
    LLM extract metadata tự động: topic, entities, date_range, category.
    """
    if LLM_API_KEY:
        try:
            return _parse_metadata(_loads_lenient(_chat(
                'Trích xuất metadata từ đoạn văn. Trả về JSON: {"topic": "...", "entities": ["..."], '
                '"category": "policy|hr|it|finance", "language": "vi|en"}',
                text, max_tokens=150, json_mode=True)))
        except Exception as e:
            print(f"  ⚠️  OpenAI metadata failed: {e}")

    return _rule_metadata(text)


def _rule_metadata(text: str) -> dict:
    """Fallback: phân loại category bằng đếm từ khóa theo nhóm."""
    lower = text.lower()
    hits = {cat: sum(lower.count(k) for k in kws) for cat, kws in _CATEGORY_KEYWORDS.items()}
    best = max(hits, key=hits.get)
    return {"topic": "general", "entities": [],
            "category": best if hits[best] > 0 else "policy", "language": "vi"}


def _parse_metadata(meta) -> dict:
    """Chỉ giữ các field mong đợi với đúng kiểu — output LLM không phải lúc nào cũng chuẩn."""
    if not isinstance(meta, dict):
        return {}
    out = {}
    if isinstance(meta.get("topic"), str):
        out["topic"] = meta["topic"]
    if isinstance(meta.get("entities"), list):
        out["entities"] = [str(e) for e in meta["entities"]]
    if meta.get("category") in ("policy", "hr", "it", "finance"):
        out["category"] = meta["category"]
    if meta.get("language") in ("vi", "en"):
        out["language"] = meta["language"]
    return out


# ─── Combined Single-Call Mode ───────────────────────────

_COMBINED_PROMPT = """Bạn chuẩn bị đoạn văn cho hệ thống tìm kiếm tài liệu nội bộ.
Dựa trên toàn bộ tài liệu (nếu có) và đoạn văn, trả về JSON:
{
  "summary": "tóm tắt 2-3 câu, giữ nguyên số liệu",
  "questions": ["3 câu hỏi nhân viên có thể hỏi mà đoạn văn trả lời được"],
  "context": "1 câu cho biết đoạn văn thuộc tài liệu nào (kèm phiên bản/năm hiệu lực nếu có) và nói về mục gì",
  "metadata": {"topic": "...", "entities": ["..."], "category": "policy|hr|it|finance", "language": "vi|en"}
}
Chỉ dùng thông tin có trong tài liệu, không suy đoán."""


def _enrich_single_call(text: str, source: str, document: str = "") -> dict:
    """Single LLM call to get summary + questions + context + metadata.

    ⚠️ Cost optimization: 1 API call thay vì 4 calls riêng lẻ.
    `document` là toàn văn tài liệu gốc — theo cách làm của Anthropic, LLM cần thấy
    cả tài liệu mới viết được câu context đúng (tên chính sách, phiên bản, mục).
    """
    if LLM_API_KEY:
        try:
            doc_part = f"<document>\n{document}\n</document>\n\n" if document else ""
            raw = _loads_lenient(_chat(_COMBINED_PROMPT,
                                   f"Tên file: {source}\n\n{doc_part}<chunk>\n{text}\n</chunk>",
                                   max_tokens=500, json_mode=True))
            questions = raw.get("questions")
            return {
                "summary": raw.get("summary") if isinstance(raw.get("summary"), str) else "",
                "questions": [str(q) for q in questions][:3] if isinstance(questions, list) else [],
                "context": raw.get("context") if isinstance(raw.get("context"), str) else "",
                "metadata": _parse_metadata(raw.get("metadata")),
            }
        except Exception as e:
            print(f"  ⚠️  Enrichment API failed: {e}")

    # Fallback không cần API: ghép từ các fallback rule-based ở trên
    title = next((ln.lstrip("#").strip() for ln in document.splitlines() if ln.startswith("# ")), "")
    return {
        "summary": _extractive_summary(text),
        "questions": [],  # câu hỏi sinh bằng rule chỉ lặp lại text → không thêm giá trị khi index
        "context": f"Trích từ tài liệu {title or source}." if (title or source) else "",
        "metadata": _rule_metadata(text),
    }


# ─── Full Enrichment Pipeline ────────────────────────────


def enrich_chunks(
    chunks: list[dict],
    methods: list[str] | None = None,
) -> list[EnrichedChunk]:
    """
    Chạy enrichment pipeline trên danh sách chunks. (Đã implement sẵn — dùng functions ở trên)

    Có 2 chế độ:
    - methods cụ thể (["summary"], ["contextual"]...): gọi từng function riêng (tốt cho học/debug)
    - methods=["combined"] hoặc None: 1 API call duy nhất cho tất cả (tốt cho production)

    Args:
        chunks: List of {"text": str, "metadata": dict}
        methods: Default None → combined mode (1 call/chunk).
                 Options: "summary", "hyqa", "contextual", "metadata", "combined"
    """
    if methods is None:
        methods = ["combined"]

    use_combined = "combined" in methods

    enriched = []
    for i, chunk in enumerate(chunks):
        text = chunk["text"]
        source = chunk.get("metadata", {}).get("source", "")

        if use_combined:
            result = _enrich_single_call(text, source, chunk.get("document", ""))
            summary = result.get("summary", "")
            questions = result.get("questions", [])
            context_line = result.get("context", "")
            enriched_text = f"{context_line}\n\n{text}" if context_line else text
            # HyQA chỉ có tác dụng khi câu hỏi được index cùng chunk
            if questions:
                enriched_text += "\n\nCâu hỏi liên quan:\n" + "\n".join(f"- {q}" for q in questions)
            auto_meta = result.get("metadata", {})
        else:
            summary = summarize_chunk(text) if "summary" in methods else ""
            questions = generate_hypothesis_questions(text) if "hyqa" in methods else []
            enriched_text = contextual_prepend(text, source) if "contextual" in methods else text
            auto_meta = extract_metadata(text) if "metadata" in methods else {}

        enriched.append(EnrichedChunk(
            original_text=text,
            enriched_text=enriched_text,
            summary=summary,
            hypothesis_questions=questions,
            auto_metadata={**chunk.get("metadata", {}), **auto_meta},
            method="+".join(methods),
        ))

        if (i + 1) % 10 == 0 or (i + 1) == len(chunks):
            print(f"  Enriched {i + 1}/{len(chunks)} chunks...", flush=True)

    return enriched


# ─── Main ────────────────────────────────────────────────

if __name__ == "__main__":
    sample = "Nhân viên chính thức được nghỉ phép năm 12 ngày làm việc mỗi năm. Số ngày nghỉ phép tăng thêm 1 ngày cho mỗi 5 năm thâm niên công tác."

    print("=== Enrichment Pipeline Demo ===\n")
    print(f"Original: {sample}\n")

    s = summarize_chunk(sample)
    print(f"Summary: {s}\n")

    qs = generate_hypothesis_questions(sample)
    print(f"HyQA questions: {qs}\n")

    ctx = contextual_prepend(sample, "Sổ tay nhân viên VinUni 2024")
    print(f"Contextual: {ctx}\n")

    meta = extract_metadata(sample)
    print(f"Auto metadata: {meta}")
