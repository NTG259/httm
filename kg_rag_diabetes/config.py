"""Cấu hình chung. Mọi đường dẫn và tham số nằm ở đây, không rải rác trong code."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW  = ROOT / "data" / "raw"          # file tải về, giữ nguyên bản gốc
PROC = ROOT / "data" / "processed"   # kết quả xử lý (chunk, triple, embedding)
DOCS = ROOT / "docs"                  # PDF tiếng Việt bổ sung (Bộ Y tế, v.v.)
RAW.mkdir(parents=True, exist_ok=True)
PROC.mkdir(parents=True, exist_ok=True)
DOCS.mkdir(parents=True, exist_ok=True)

# --- Nguồn dữ liệu -------------------------------------------------------
HETIONET_BASE = "https://github.com/hetio/hetionet/raw/v1.0.0/hetnet/tsv/"
HETIONET_NODES = "hetionet-v1.0-nodes.tsv"
HETIONET_EDGES = "hetionet-v1.0-edges.sif.gz"

MEDLINEPLUS_INDEX = "https://medlineplus.gov/xml.html"   # trang liệt kê file XML theo ngày

WHO_PDF_URL = ("https://iris.who.int/server/api/core/bitstreams/"
               "d2997184-51c3-4b0b-aa2c-0cc840424b6c/content")
WHO_PDF_NAME = "who_global_report_on_diabetes_2016.pdf"

HEADERS = {"User-Agent": "kg-rag-student-demo/1.0 (educational use)"}

# --- Tham số xử lý văn bản ----------------------------------------------
# all-MiniLM-L6-v2 chỉ đọc tối đa 256 token, nên chunk quá dài sẽ bị cắt khi tạo embedding.
CHUNK_WORDS = 150
CHUNK_OVERLAP = 30

# --- Tham số truy xuất ---------------------------------------------------
EMBED_BACKEND = os.getenv("EMBED_BACKEND", "sbert")   # "sbert" hoặc "tfidf" (phương án dự phòng)
SBERT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 3
# Ngưỡng cosine tối thiểu để coi là "có bằng chứng". Đây chỉ là giá trị khởi đầu:
# hãy in điểm của các câu hỏi không có đáp án (demo_b.py) rồi chỉnh lại cho đúng dữ liệu của bạn.
MIN_SCORE = float(os.getenv("MIN_SCORE", "0.30"))

# --- Tham số đồ thị ------------------------------------------------------
KEEP_METAEDGES = {"CtD", "CpD", "DpS", "DrD", "DlA", "DaG"}
MAX_FACTS_PER_RELATION = 8     # không đổ cả đồ thị vào ngữ cảnh (Phần 7: graph retrieval quá rộng)
