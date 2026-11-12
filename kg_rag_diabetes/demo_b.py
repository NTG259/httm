"""Demo phần B. Chạy mặc định 5 câu hỏi, hoặc: python demo_b.py "câu hỏi của bạn"."""
import sys

from hybrid import build_prompt, format_context, hybrid_context

QUESTIONS = [
    "Which drugs treat type 2 diabetes mellitus?",                     # thiên về quan hệ (đồ thị)
    "How many adults were living with diabetes worldwide in 2014?",    # thiên về văn bản (vector)
    "What symptoms are associated with type 2 diabetes mellitus, and what complications does diabetes cause?",  # hybrid
    "Who won the 2022 FIFA World Cup?",                                # ngoài phạm vi: ngưỡng điểm phải chặn
    "What is the recommended insulin dose for a 12-year-old child?",   # ca khó: có chủ đề, nhưng không có liều trong kho
]


def show(question: str, with_prompt: bool = False) -> None:
    ctx = hybrid_context(question)
    print("=" * 78)
    print("QUESTION:", question)
    print("Entities:", ctx["entities"] or "(none)")
    print(format_context(ctx))
    print("Top-k cosine:", [round(d["score"], 3) for d in ctx["docs"]])
    print("Đủ bằng chứng?", "CÓ" if ctx["enough"] else "KHÔNG -> hệ thống phải trả lời 'không đủ thông tin'")
    if with_prompt:
        print("-" * 78)
        print(build_prompt(question, format_context(ctx)))


if __name__ == "__main__":
    questions = sys.argv[1:] or QUESTIONS
    for q in questions:
        show(q, with_prompt=bool(sys.argv[1:]))
