from __future__ import annotations

"""Module 4: RAGAS Evaluation — 4 metrics + failure analysis."""

import os, sys, json
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (LLM_API_KEY, LLM_BASE_URL, LLM_EMBEDDING_MODEL, LLM_MAX_RETRIES,
                    LLM_MODEL, TEST_SET_PATH)

METRICS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]


@dataclass
class EvalResult:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


def load_test_set(path: str = TEST_SET_PATH) -> list[dict]:
    """Load test set from JSON. (Đã implement sẵn)"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def evaluate_ragas(questions: list[str], answers: list[str],
                   contexts: list[list[str]], ground_truths: list[str]) -> dict:
    """Run RAGAS evaluation.

    Trả về điểm trung bình của 4 metric + "per_question" (list[EvalResult]).
    RAGAS cho NaN khi judge LLM trả output không parse được → trung bình bỏ qua NaN
    thay vì kéo cả metric về NaN/0.
    """
    zeros = {m: 0.0 for m in METRICS}
    if not questions:
        return {**zeros, "per_question": []}
    if not LLM_API_KEY:
        print("  ⚠️  RAGAS evaluation skipped: thiếu OPENAI_API_KEY / GEMINI_API_KEY (RAGAS cần LLM làm judge).")
        return {**zeros, "per_question": []}

    try:
        import math
        from datasets import Dataset
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        from ragas import evaluate
        from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness

        dataset = Dataset.from_dict({
            "question": questions, "answer": answers,
            "contexts": contexts, "ground_truth": ground_truths,
        })
        from ragas.run_config import RunConfig

        # Mặc định ragas 0.1.x dùng gpt-3.5-turbo + ada-002 → chỉ định model rõ ràng,
        # temperature=0 để điểm ổn định giữa các lần chạy.
        judge = ChatOpenAI(model=LLM_MODEL, temperature=0, api_key=LLM_API_KEY,
                           base_url=LLM_BASE_URL, max_retries=LLM_MAX_RETRIES)
        # check_embedding_ctx_length=False: gửi text thô thay vì token id của tiktoken
        # (endpoint không phải OpenAI, như Gemini, không hiểu token id).
        embeddings = OpenAIEmbeddings(model=LLM_EMBEDDING_MODEL, api_key=LLM_API_KEY,
                                      base_url=LLM_BASE_URL, check_embedding_ctx_length=False)
        # answer_relevancy mặc định sinh 3 câu hỏi ngược bằng 1 request n=3; Gemini
        # không hỗ trợ n>1 → giảm strictness về 1 (vẫn cùng công thức, ít mẫu hơn).
        answer_relevancy.strictness = 3 if LLM_BASE_URL is None else 1

        result = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
            llm=judge,
            embeddings=embeddings,
            raise_exceptions=False,
            # Ít worker + retry dài để không vỡ rate limit của free tier
            run_config=RunConfig(max_workers=4, max_retries=10, max_wait=90, timeout=300),
        )
        df = result.to_pandas()

        def _f(v) -> float:
            try:
                return float(v)
            except (TypeError, ValueError):
                return float("nan")

        per_question = [
            EvalResult(question=q, answer=a, contexts=list(c), ground_truth=gt,
                       **{m: _f(row.get(m)) for m in METRICS})
            for (q, a, c, gt), (_, row) in zip(zip(questions, answers, contexts, ground_truths),
                                                df.iterrows())
        ]

        def _mean(m: str) -> float:
            vals = [getattr(r, m) for r in per_question if not math.isnan(getattr(r, m))]
            return sum(vals) / len(vals) if vals else 0.0

        return {**{m: _mean(m) for m in METRICS}, "per_question": per_question}
    except Exception as e:
        print(f"  ⚠️  RAGAS evaluation failed: {e}")
        return {**zeros, "per_question": []}


# Diagnostic Tree: metric thấp nhất → (chẩn đoán, hướng xử lý)
DIAGNOSTIC_TREE = {
    "faithfulness": ("LLM tự bịa câu trả lời ngoài tài liệu (hallucination)",
                     "Thắt chặt system prompt, giảm temperature về 0"),
    "context_recall": ("Hệ thống tìm kiếm bỏ sót đoạn văn chứa đáp án",
                       "Cải thiện chunking hoặc bổ sung từ khóa BM25"),
    "context_precision": ("Đoạn văn không liên quan bị xếp lên đầu",
                          "Bổ sung cross-encoder reranking hoặc lọc theo metadata"),
    "answer_relevancy": ("Câu trả lời lệch trọng tâm câu hỏi",
                         "Viết lại prompt hướng dẫn mô hình trả lời trực tiếp hơn"),
}


def failure_analysis(eval_results: list[EvalResult], bottom_n: int = 10) -> list[dict]:
    """Analyze bottom-N worst questions using Diagnostic Tree.

    Xếp câu hỏi theo điểm trung bình 4 metric tăng dần; với mỗi câu lấy metric thấp
    nhất để tra chẩn đoán. Metric NaN (judge lỗi) bị bỏ qua, không coi là 0.
    """
    import math

    analyzed = []
    for r in eval_results:
        scores = {m: getattr(r, m) for m in METRICS}
        valid = {m: s for m, s in scores.items() if s is not None and not math.isnan(s)}
        if not valid:
            continue
        worst_metric = min(valid, key=valid.get)
        diagnosis, fix = DIAGNOSTIC_TREE[worst_metric]
        analyzed.append({
            "question": r.question,
            "avg_score": round(sum(valid.values()) / len(valid), 4),
            "worst_metric": worst_metric,
            "score": round(valid[worst_metric], 4),
            "diagnosis": diagnosis,
            "suggested_fix": fix,
            "scores": {m: (round(s, 4) if m in valid else None) for m, s in scores.items()},
            "answer": r.answer,
            "ground_truth": r.ground_truth,
            "contexts": r.contexts,
        })

    analyzed.sort(key=lambda x: x["avg_score"])
    return analyzed[:bottom_n]


def save_report(results: dict, failures: list[dict], path: str = "reports/ragas_report.json"):
    """Save evaluation report to JSON. (Đã implement sẵn)"""
    parent_dir = os.path.dirname(path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    import math
    from dataclasses import asdict

    def _clean(v):  # NaN không hợp lệ trong JSON chuẩn
        return None if isinstance(v, float) and math.isnan(v) else v

    per_question = [{k: _clean(v) for k, v in asdict(r).items()}
                    for r in results.get("per_question", [])]
    report = {
        "aggregate": {k: v for k, v in results.items() if k != "per_question"},
        "num_questions": len(per_question),
        "failures": failures,
        "per_question": per_question,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Report saved to {path}")


if __name__ == "__main__":
    test_set = load_test_set()
    print(f"Loaded {len(test_set)} test questions")
    print("Run pipeline.py first to generate answers, then call evaluate_ragas().")
