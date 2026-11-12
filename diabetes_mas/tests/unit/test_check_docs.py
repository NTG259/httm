import shutil

import pytest

from diabetes_mas.config import PROJECT_ROOT
from diabetes_mas.devtools.check_docs import check

DOCS = PROJECT_ROOT.parent / "docs"
pytestmark = pytest.mark.skipif(not (DOCS / "01_REQUIREMENTS.md").exists(), reason="thiếu thư mục docs")


def test_real_docs_are_fully_traced():
    assert check(DOCS) == []


def test_detects_missing_trace_row_and_question(tmp_path):
    mut = tmp_path / "docs"
    shutil.copytree(DOCS, mut)
    req = mut / "01_REQUIREMENTS.md"
    text = req.read_text(encoding="utf-8")
    text = "\n".join(l for l in text.splitlines() if not l.startswith("| TK-17 | FR-17"))
    text = text.replace("| Q46 |", "| Q99 |")
    req.write_text(text, encoding="utf-8")
    errs = check(mut)
    assert any("TK-17" in e for e in errs)
    assert any("Q46" in e for e in errs)


def test_sequence_check_flags_unpermitted_tool_and_missing_channel(tmp_path):
    """[9] GĐ 4: mũi tên gọi tool mà agent không được cấp, hoặc agent→agent không ghi kênh, phải bị bắt."""
    mut = tmp_path / "docs"
    shutil.copytree(DOCS, mut)
    seq = mut / "04_SEQUENCE.md"
    text = seq.read_text(encoding="utf-8")
    old = "  N-->>ST: tuoi = 52, chieu_cao, can_nang, gioi_tinh = nam (K1)\n"
    assert old in text                                # sơ đồ đổi thì sửa test, không để đột biến rơi vào khoảng trống
    text = text.replace(old, "  N->>ST: T-05 update_state\n", 1)
    text = text.replace("M-->>CL: CLASSIFY (K1)", "M-->>CL: CLASSIFY", 1)
    text = text.replace("participant RG as AG-11 Reward", "participant RG as AG-11 Reward\n  participant SF as AG-08 Safety", 1)
    seq.write_text(text, encoding="utf-8")
    errs = check(mut)
    assert any("AG-06 gọi T-05" in e for e in errs)
    assert any("thiếu tool/kênh" in e and "CLASSIFY" in e for e in errs)
    assert any("AG-08 không chạy ở chế độ 'train'" in e for e in errs)
