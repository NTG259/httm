"""Làm sạch dữ liệu thô → data/interim/docs.jsonl + data/processed/sources.jsonl.

Chạy:  python -m diabetes_mas.ingest.clean

Mỗi dòng docs.jsonl là một **mục** (section) của một nguồn:
  doc_id, source_id, title, section, text, lang, page
"""
from __future__ import annotations

import re
import sys

from pypdf import PdfReader

from diabetes_mas.config import INTERIM_DIR, PROCESSED_DIR, RAW_DIR, load_yaml
from diabetes_mas.ingest.parse_vinmec import parse
from diabetes_mas.utils.text import collapse_ws, read_jsonl, write_jsonl

DOCS = INTERIM_DIR / "docs.jsonl"
SOURCES = PROCESSED_DIR / "sources.jsonl"

# Tiêu đề mục trong văn bản Bộ Y tế: "I. KHÁI NIỆM", "IV. ĐIỀU TRỊ ..."
MOH_HEADING = re.compile(r"^(?:[IVX]{1,4})\.\s+[A-ZÀ-Ỹ][^\n]{3,120}$", re.M)
# Dòng ghi người viết/tư vấn chuyên môn trong bài Vinmec → tín hiệu kiểm duyệt thật
REVIEWER = re.compile(r"Bài viết (?:được )?(?:viết|tư vấn chuyên môn|tham vấn(?: y khoa)?) bởi\s+([^\n]{3,160})")


def find_reviewer(texts: list[str]) -> str | None:
    for t in texts[:3]:
        m = REVIEWER.search(t)
        if m:
            return m.group(1).strip().rstrip(".")
    return None


def vinmec_docs() -> tuple[list[dict], list[dict]]:
    docs, sources = [], []
    for row in read_jsonl(RAW_DIR / "vinmec" / "index.jsonl"):
        html = (RAW_DIR / row["file"]).read_text(encoding="utf-8")
        page = parse(html, row["kind"])
        if not page.sections:
            print(f"[rỗng] {row['source_id']} {row['url']}", file=sys.stderr)
            continue
        reviewer = find_reviewer([s.text for s in page.sections])
        sources.append({
            "id": row["source_id"], "publisher": "Vinmec", "title": page.title, "url": row["url"],
            "published": (page.published or "")[:10] or None, "author": reviewer or page.author,
            # True khi bài ghi rõ bác sĩ viết/tư vấn; None = không rõ (KHÔNG mặc định là False)
            "reviewed": True if reviewer else None, "lang": "vi", "kind": f"vinmec_{row['kind']}",
        })
        for i, s in enumerate(page.sections):
            docs.append({
                "doc_id": f"{row['source_id']}#s{i:02d}", "source_id": row["source_id"], "title": page.title,
                "section": s.heading, "text": s.text, "lang": "vi", "page": None,
            })
    return docs, sources


def _pdf_pages(path) -> list[str]:
    reader = PdfReader(str(path))
    return [(p.extract_text() or "") for p in reader.pages]


def pdf_docs(entry: dict, lang: str) -> list[dict]:
    """Mỗi trang PDF → 1 doc; section = tiêu đề La Mã gần nhất (với văn bản Bộ Y tế)."""
    path = RAW_DIR / entry["dest"]
    docs, section = [], entry["title"]
    for pno, raw in enumerate(_pdf_pages(path), start=1):
        headings = MOH_HEADING.findall(raw) if lang == "vi" else []
        text = "\n".join(collapse_ws(line) for line in raw.splitlines() if line.strip())
        if len(text) < 80:
            continue
        docs.append({
            "doc_id": f"{entry['id']}#p{pno:03d}", "source_id": entry["id"], "title": entry["title"],
            "section": section if not headings else collapse_ws(headings[0]),
            "text": text, "lang": lang, "page": pno,
        })
        if headings:
            section = collapse_ws(headings[-1])
    return docs


def main() -> int:
    docs, sources = vinmec_docs()
    for e in load_yaml("sources.yaml")["static"]:
        if not (RAW_DIR / e["dest"]).exists():
            print(f"[thiếu file] {e['id']} — chạy fetch_static trước", file=sys.stderr)
            continue
        sources.append({
            "id": e["id"], "publisher": e["publisher"], "title": e["title"], "url": e.get("url"),
            "published": e.get("published"), "author": None, "reviewed": e.get("reviewed"),
            "lang": "en" if e["publisher"] in {"WHO", "Hetionet"} else "vi", "kind": e["kind"],
        })
        if e["kind"] == "pdf":
            docs.extend(pdf_docs(e, lang=sources[-1]["lang"]))
    docs.sort(key=lambda d: d["doc_id"])
    sources.sort(key=lambda s: s["id"])
    n_docs = write_jsonl(DOCS, docs)
    n_src = write_jsonl(SOURCES, sources)
    print(f"[clean] {n_src} nguồn, {n_docs} mục → {DOCS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
