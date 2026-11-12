"""Bước 2: đọc tài liệu thật -> làm sạch -> cắt chunk -> lưu data/processed/chunks.jsonl
Mỗi chunk giữ metadata (nguồn, trang hoặc url) để trích dẫn về sau."""
import html
import json
import re
import xml.etree.ElementTree as ET

from pypdf import PdfReader

from config import CHUNK_OVERLAP, CHUNK_WORDS, DOCS, PROC, RAW, WHO_PDF_NAME


def clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def strip_html(text: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", " ", html.unescape(text)))


def word_chunks(text: str, size: int = CHUNK_WORDS, overlap: int = CHUNK_OVERLAP):
    words = text.split()
    step = max(size - overlap, 1)
    for start in range(0, len(words), step):
        piece = words[start:start + size]
        if piece:
            yield " ".join(piece)
        if start + size >= len(words):
            break


def who_chunks(pdf_path):
    reader = PdfReader(str(pdf_path))
    for page_no, page in enumerate(reader.pages, start=1):   # số trang của file PDF, không phải số in trên giấy
        text = clean(page.extract_text() or "")
        if len(text.split()) < 30:       # bỏ trang gần như trống (bìa, trang phân cách)
            continue
        for i, chunk in enumerate(word_chunks(text)):
            yield {"id": f"who:p{page_no}:{i}", "source": "WHO Global report on diabetes (2016)",
                   "page": page_no, "url": "https://www.who.int/publications/i/item/9789241565257",
                   "text": chunk}


def medlineplus_chunks(xml_path):
    root = ET.parse(xml_path).getroot()
    total = hit = 0
    for topic in root.iter("health-topic"):
        if (topic.get("language") or "English") != "English":
            continue
        total += 1
        title = topic.get("title") or topic.findtext("title") or ""
        names = [title] + [e.text or "" for e in topic.findall("also-called")]
        names += [d.text or "" for d in topic.iter("descriptor")]
        if not any("diabet" in n.lower() for n in names):
            continue
        hit += 1
        summary = clean(strip_html(topic.findtext("full-summary") or ""))
        for i, chunk in enumerate(word_chunks(summary)):
            yield {"id": f"mplus:{topic.get('id')}:{i}", "source": f"MedlinePlus: {title}",
                   "page": None, "url": topic.get("url"), "text": chunk}
    print(f"[MedlinePlus] {hit}/{total} chủ đề tiếng Anh liên quan tới diabetes")


# Tên hiển thị cho các PDF tiếng Việt đã biết; file không có trong dict sẽ dùng stem làm tên.
_VI_PDF_NAMES: dict[str, str] = {
    "qd5481_dtd_tip2_2020":   "Bộ Y tế – Hướng dẫn chẩn đoán và điều trị ĐTĐ típ 2 (QĐ 5481/2020)",
    "hd_dtd_tip1_2024":       "Bộ Y tế – Hướng dẫn chẩn đoán và điều trị ĐTĐ típ 1 (2024)",
    "qd3087_tien_dtd_2020":   "Bộ Y tế – Hướng dẫn chẩn đoán và điều trị Tiền ĐTĐ (QĐ 3087/2020)",
}


def vi_pdf_chunks(pdf_path):
    """Đọc PDF tiếng Việt từ thư mục docs/ và tạo chunk giống who_chunks()."""
    stem = pdf_path.stem
    source = _VI_PDF_NAMES.get(stem, stem.replace("_", " "))
    reader = PdfReader(str(pdf_path))
    for page_no, page in enumerate(reader.pages, start=1):
        text = clean(page.extract_text() or "")
        if len(text.split()) < 20:
            continue
        for i, chunk in enumerate(word_chunks(text)):
            yield {
                "id":     f"vi:{stem}:p{page_no}:{i}",
                "source": source,
                "page":   page_no,
                "url":    None,
                "text":   chunk,
            }


def main() -> None:
    chunks = []
    pdf = RAW / WHO_PDF_NAME
    if pdf.exists():
        chunks += list(who_chunks(pdf))
    else:
        print(f"[cảnh báo] thiếu {pdf.name}, bỏ qua nguồn WHO")
    xmls = sorted(RAW.glob("mplus_topics_*.xml"))
    if xmls:
        chunks += list(medlineplus_chunks(xmls[-1]))      # file mới nhất theo ngày trong tên
    else:
        print("[cảnh báo] thiếu file mplus_topics_*.xml, bỏ qua nguồn MedlinePlus")
    vi_pdfs = sorted(DOCS.glob("*.pdf"))
    if vi_pdfs:
        for pdf in vi_pdfs:
            vi_chunks = list(vi_pdf_chunks(pdf))
            chunks += vi_chunks
            print(f"[VI] {pdf.name}: {len(vi_chunks)} chunk")
    else:
        print("[cảnh báo] không có PDF tiếng Việt trong docs/, bỏ qua")

    if not chunks:
        raise SystemExit("Không có chunk nào. Chạy download_data.py trước.")
    with open(PROC / "chunks.jsonl", "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    by_source = {}
    for c in chunks:
        key = c["source"].split(":")[0]
        by_source[key] = by_source.get(key, 0) + 1
    print(f"[xong] {len(chunks)} chunk -> {PROC / 'chunks.jsonl'}  {by_source}")


if __name__ == "__main__":
    main()
