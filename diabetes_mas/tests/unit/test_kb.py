import pytest

from diabetes_mas.kb.build_cypher import lit
from diabetes_mas.kb.link import Term, link_text
from diabetes_mas.kb.seed import evidence_found


def test_lit_escapes_quotes_backslash_newline():
    assert lit("a'b\\c\nd") == "'a\\'b\\\\c\\nd'"
    assert lit({"b": 1, "a": [True, None, 1.5]}) == "{a: [true, null, 1.5], b: 1}"


def test_lit_rejects_nan():
    with pytest.raises(ValueError):
        lit(float("nan"))


def test_link_prefers_longest_term_and_ignores_accents():
    lex = sorted([
        Term("tieu duong", "DTD", "Disease"),
        Term("tieu duong tuyp 2", "DTD_TIP_2", "Disease"),
    ], key=lambda t: -len(t.key))
    found = link_text("Bệnh Tiểu đường tuýp 2 và tiểu đường nói chung", lex)
    assert found == {("DTD_TIP_2", "Disease"): 1, ("DTD", "Disease"): 1}


def test_link_word_boundary():
    lex = [Term("insulin", "INSULIN", "Drug")]
    assert link_text("kháng insulinoma", lex) == {}


@pytest.mark.skipif(not evidence_found("MOH-3280-2011", "Tuổi ≥ 45"), reason="cần chạy ingest.clean trước")
def test_evidence_check_rejects_fabricated_quote():
    assert evidence_found("MOH-3280-2011", "Tuổi ≥ 45")
    assert not evidence_found("MOH-3280-2011", "HbA1c ≥ 6,5% là tiêu chuẩn chẩn đoán")
    assert not evidence_found("VMC-0002", "Tuổi ≥ 45")          # đúng câu nhưng sai nguồn
