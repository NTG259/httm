from pathlib import Path

from diabetes_mas.ingest.parse_vinmec import parse_disease, parse_post

FIX = Path(__file__).resolve().parents[1] / "fixtures"


def _all_text(page):
    return "\n".join(s.text for s in page.sections)


def test_post_inline_tags_do_not_split_words():
    page = parse_post((FIX / "vinmec_post.html").read_text(encoding="utf-8"))
    text = _all_text(page)
    assert "tiền sản giật" in text
    assert "tiền sản giậ\n" not in text


def test_post_metadata_and_sections():
    page = parse_post((FIX / "vinmec_post.html").read_text(encoding="utf-8"))
    assert page.title.startswith("Đái tháo đường thai kỳ")
    assert page.published and page.published.startswith("2024-07-07")
    headings = [s.heading for s in page.sections]
    assert any("Cách phát hiện và chẩn đoán" in h for h in headings)


def test_post_drops_toc_and_cta():
    page = parse_post((FIX / "vinmec_post.html").read_text(encoding="utf-8"))
    text = _all_text(page)
    assert "Mục lục" not in text
    assert "HOTLINE" not in text
    assert "My Vinmec" not in text


def test_disease_sections_and_no_duplicate_lines():
    page = parse_disease((FIX / "vinmec_disease.html").read_text(encoding="utf-8"))
    assert page.title.startswith("Tiểu đường tuýp 2")
    diag = [s for s in page.sections if s.heading.startswith("Các biện pháp chẩn đoán")]
    assert diag, [s.heading for s in page.sections]
    text = "\n".join(s.text for s in diag)
    assert "HbA1C (định lượng bằng phương pháp sắc kí lỏng) >6,5%" in text
    assert text.count("HbA1C (định lượng") == 1
