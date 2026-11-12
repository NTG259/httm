"""Bước 1: tải dữ liệu thật về data/raw. Chạy lại an toàn: file đã có thì bỏ qua."""
import re
from pathlib import Path
from urllib.parse import urljoin

import requests

from config import (HEADERS, HETIONET_BASE, HETIONET_EDGES, HETIONET_NODES,
                    MEDLINEPLUS_INDEX, RAW, WHO_PDF_NAME, WHO_PDF_URL)


def fetch(url: str, dest: Path, must_start_with: bytes | None = None) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"[bỏ qua] {dest.name} đã có")
        return
    print(f"[tải]    {url}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, headers=HEADERS, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for block in r.iter_content(1 << 20):
                f.write(block)
    if must_start_with:
        with open(tmp, "rb") as f:
            if not f.read(len(must_start_with)).startswith(must_start_with):
                tmp.unlink()
                raise RuntimeError(f"{dest.name}: nội dung tải về không đúng định dạng")
    tmp.rename(dest)
    print(f"[xong]   {dest.name}: {dest.stat().st_size / 1e6:.1f} MB")


def latest_medlineplus_url() -> str:
    """Không ghi cứng ngày: đọc trang xml.html rồi lấy file mplus_topics_YYYY-MM-DD.xml mới nhất."""
    html = requests.get(MEDLINEPLUS_INDEX, headers=HEADERS, timeout=60).text
    found = re.findall(r'href="([^"]*mplus_topics_(\d{4}-\d{2}-\d{2})\.xml)"', html)
    if not found:
        raise RuntimeError("Không tìm thấy link mplus_topics_*.xml trên trang xml.html")
    href, _date = max(found, key=lambda x: x[1])
    return urljoin(MEDLINEPLUS_INDEX, href)


def main() -> None:
    # 1) Đồ thị tri thức: Hetionet (CC0), bản v1.0.0 cố định để kết quả lặp lại được
    fetch(HETIONET_BASE + HETIONET_NODES, RAW / HETIONET_NODES)
    fetch(HETIONET_BASE + HETIONET_EDGES, RAW / HETIONET_EDGES, must_start_with=b"\x1f\x8b")

    # 2) Văn bản: MedlinePlus Health Topics XML (bản mới nhất)
    url = latest_medlineplus_url()
    fetch(url, RAW / Path(url).name)

    # 3) Văn bản: báo cáo WHO (PDF). Nếu tải tự động lỗi thì tải tay vào đúng tên file.
    try:
        fetch(WHO_PDF_URL, RAW / WHO_PDF_NAME, must_start_with=b"%PDF")
    except Exception as e:  # noqa: BLE001
        print(f"[lỗi] Không tải được PDF WHO: {e}")
        print("      Tải tay từ https://www.who.int/publications/i/item/9789241565257")
        print(f"      rồi lưu vào {RAW / WHO_PDF_NAME}")


if __name__ == "__main__":
    main()
