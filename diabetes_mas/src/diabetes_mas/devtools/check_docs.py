"""Kiểm tra truy vết giữa các tài liệu (tiêu chí hoàn thành Giai đoạn 1).

Chạy:  python -m diabetes_mas.devtools.check_docs [--docs ../docs]

Kiểm tra:
 1. Mọi TK ở §5 của 01 có dòng trong ma trận truy vết §12, với FR và Agent không rỗng.
 2. FR trong ma trận tồn tại ở §6; Agent tồn tại trong 02_AGENT_DESIGN.md; TC tồn tại trong 07_TEST_PLAN.md.
 3. Mọi FR "Must" có ≥ 1 test case trong 07 có cột FR chứa nó.
 4. Mọi câu Q19–Q51 có trong ma trận phủ §8 với cột artifact không rỗng.
 5. Mọi FR trạng thái ✅ có cột bằng chứng không rỗng.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from diabetes_mas.config import PROJECT_ROOT

ID_TK = re.compile(r"TK-\d+")
ID_FR = re.compile(r"FR-\d+")
ID_TC = re.compile(r"TC-\d+[a-z]?")
ID_AG = re.compile(r"AG-(?:W\d|\d+(?:-[\dh])?)")
ID_Q = re.compile(r"Q(\d+)")


def sections(text: str) -> dict[str, str]:
    """'## 5. Danh sách…' → key '5'."""
    out: dict[str, str] = {}
    for block in re.split(r"(?m)^## ", text)[1:]:
        m = re.match(r"(\d+[a-z]?)\.", block)
        if m:
            out[m.group(1)] = block
    return out


def rows(block: str) -> list[list[str]]:
    res = []
    for line in block.splitlines():
        line = line.strip()
        if not line.startswith("|") or re.match(r"^\|[\s:|-]+\|$", line):
            continue
        res.append([c.strip() for c in line.strip("|").split("|")])
    return res


def check(docs: Path) -> list[str]:
    errs: list[str] = []
    req = sections((docs / "01_REQUIREMENTS.md").read_text(encoding="utf-8"))
    agent_doc = (docs / "02_AGENT_DESIGN.md").read_text(encoding="utf-8")
    test_doc = (docs / "07_TEST_PLAN.md").read_text(encoding="utf-8")

    tasks = {r[0] for r in rows(req["5"]) if ID_TK.fullmatch(r[0])}
    frs = {r[0]: r for r in rows(req["6"]) if ID_FR.fullmatch(r[0])}
    agents_known = set(ID_AG.findall(agent_doc))
    tc_rows = [r for r in rows(test_doc) if r and ID_TC.fullmatch(r[0])]
    tcs_known = {r[0] for r in tc_rows}

    trace = {r[0]: r for r in rows(req["12"]) if ID_TK.fullmatch(r[0])}
    for tk in sorted(tasks, key=lambda x: int(x[3:])):
        if tk not in trace:
            errs.append(f"[1] {tk} thiếu trong ma trận truy vết §12")
            continue
        _, fr_cell, ag_cell, tc_cell = (trace[tk] + ["", "", ""])[:4]
        if not ID_FR.findall(fr_cell):
            errs.append(f"[1] {tk}: cột FR rỗng")
        if not ag_cell:
            errs.append(f"[1] {tk}: cột Agent rỗng (ghi '—' nếu không có agent)")
        for fr in ID_FR.findall(fr_cell):
            if fr not in frs:
                errs.append(f"[2] {tk}: {fr} không có trong §6")
        for ag in ID_AG.findall(ag_cell):
            if ag not in agents_known:
                errs.append(f"[2] {tk}: {ag} không có trong 02_AGENT_DESIGN.md")
        for tc in ID_TC.findall(tc_cell):
            if tc not in tcs_known:
                errs.append(f"[2] {tk}: {tc} không có trong 07_TEST_PLAN.md")

    for fr, r in frs.items():
        priority = r[3] if len(r) > 3 else ""
        if priority == "Must":
            covering = [t[0] for t in tc_rows if fr in ID_FR.findall(t[-1])]
            if not covering:
                errs.append(f"[3] {fr} (Must) chưa có test case nào trong 07_TEST_PLAN.md")
        if len(r) > 5 and "✅" in r[4] and not r[5]:
            errs.append(f"[5] {fr} đánh dấu ✅ nhưng thiếu bằng chứng")

    errs += check_registry(docs, agent_doc)

    covered = {}
    for r in rows(req["8"]):
        for q in ID_Q.findall(r[0]):
            covered[int(q)] = r
        m = re.match(r"Q(\d+)[–-]Q(\d+)", r[0])
        if m:
            for q in range(int(m.group(1)), int(m.group(2)) + 1):
                covered[q] = r
    for q in range(19, 52):
        if q not in covered:
            errs.append(f"[4] Q{q} thiếu trong ma trận phủ §8")
        elif len(covered[q]) < 3 or not covered[q][2]:
            errs.append(f"[4] Q{q}: chưa ghi artifact trả lời")
    return errs


ID_T = re.compile(r"T-\d+")


def _expand_agents(cell: str) -> set[str]:
    out = set(ID_AG.findall(cell))
    if "AG-W*" in cell or "AG-W1..W4" in cell:
        out |= {"AG-W1", "AG-W2", "AG-W3", "AG-W4"}
    return out


def check_registry(docs: Path, agent_doc: str) -> list[str]:
    """[6] configs/agents.yaml ↔ thẻ agent trong 02 §2; [7] quyền tool ↔ bảng tổng hợp 03 §2."""
    from diabetes_mas.agents.registry import agents

    errs: list[str] = []
    reg = agents()
    cards = {r[0] for r in rows(sections(agent_doc).get("2", "")) if r and ID_AG.fullmatch(r[0])}
    for a in sorted(set(reg) - cards):
        errs.append(f"[6] {a} có trong agents.yaml nhưng thiếu thẻ ở 02 §2")
    for a in sorted(cards - set(reg)):
        errs.append(f"[6] {a} có thẻ ở 02 §2 nhưng thiếu trong agents.yaml")

    tool_doc = sections((docs / "03_TOOL_SPEC.md").read_text(encoding="utf-8")).get("2", "")
    doc_pairs: set[tuple[str, str]] = set()
    tools_in_doc: set[str] = set()
    for r in rows(tool_doc):
        if r and ID_T.fullmatch(r[0]) and len(r) >= 4:
            tools_in_doc.add(r[0])
            for a in _expand_agents(r[3]):
                doc_pairs.add((a, r[0]))
    reg_pairs = {(a, t) for a, spec in reg.items() for t in spec["tools"]}
    for a, t in sorted(reg_pairs):
        if t not in tools_in_doc:
            errs.append(f"[7] {a} được cấp {t} nhưng {t} không có trong 03 §2")
        elif t != "T-06" and (a, t) not in doc_pairs:
            errs.append(f"[7] {a} được cấp {t} trong agents.yaml nhưng 03 §2 không ghi {a} ở cột agent")
    for a, t in sorted(doc_pairs - reg_pairs):
        if a in reg:
            errs.append(f"[7] 03 §2 ghi {a} dùng {t} nhưng agents.yaml không cấp")

    # [8] tool trong tài liệu ↔ tool đã cài đặt (đăng ký trong tools.TOOLS)
    from diabetes_mas.tools import TOOLS

    for t in sorted(tools_in_doc - set(TOOLS), key=lambda x: int(x[2:])):
        errs.append(f"[8] {t} có trong 03 §2 nhưng chưa có cài đặt (tools.TOOLS)")
    for t in sorted(set(TOOLS) - tools_in_doc, key=lambda x: int(x[2:])):
        errs.append(f"[8] {t} đã cài đặt nhưng thiếu trong 03 §2")
    for r in rows(tool_doc):
        if r and r[0] in TOOLS and f"`{TOOLS[r[0]].name}`" != r[1]:
            errs.append(f"[8] {r[0]}: tên trong 03 ({r[1]}) khác tên cài đặt ({TOOLS[r[0]].name})")
    seq = docs / "04_SEQUENCE.md"
    if seq.exists():
        errs += check_sequences(seq.read_text(encoding="utf-8"), reg, set(TOOLS))
    return errs


MERMAID = re.compile(r"```mermaid\n(.*?)```", re.S)
PARTICIPANT = re.compile(r"^\s*(?:participant|actor)\s+(\w+)(?:\s+as\s+(.*))?$")
ARROW = re.compile(r"^\s*(\w+)\s*(?:--?>>|--?\)|--?x)\s*([+-]?)(\w+)\s*:\s*(.*)$")
CHANNEL = re.compile(r"\bK\d\b")
MODE = re.compile(r"%%\s*mode:\s*(\w+)")


def _participant_agents(label: str) -> set[str]:
    if "AG-Wh" in label:
        return {"AG-W1", "AG-W2", "AG-W3", "AG-W4"}
    return set(ID_AG.findall(label))


def check_sequences(text: str, reg: dict, tools: set[str]) -> list[str]:
    """[9] 04_SEQUENCE.md — tiêu chí hoàn thành GĐ 4, mục 1:
    - mũi tên ghi T-xx: tool tồn tại, và agent gửi được cấp tool đó trong agents.yaml;
    - mũi tên giữa hai agent khác nhau mà không gọi tool: phải ghi kênh K1..K6;
    - khối có `%% mode: <m>`: mọi agent trong sơ đồ phải chạy được ở chế độ đó."""
    errs: list[str] = []
    for n, block in enumerate(MERMAID.findall(text), 1):
        if "sequenceDiagram" not in block:
            continue
        who: dict[str, set[str]] = {}
        mode = (MODE.search(block) or [None, None])[1]
        for line in block.splitlines():
            m = PARTICIPANT.match(line)
            if m:
                who[m.group(1)] = _participant_agents(m.group(2) or "")
                if mode:
                    for a in who[m.group(1)]:
                        if a in reg and mode not in reg[a]["modes"]:
                            errs.append(f"[9] sơ đồ {n}: {a} không chạy ở chế độ '{mode}'")
                continue
            m = ARROW.match(line)
            if not m:
                continue
            src, _, dst, label = m.groups()
            src_ag, dst_ag = who.get(src, set()), who.get(dst, set())
            called = [t for t in ID_T.findall(label) if t != "T-10"]
            for t in called:
                if t not in tools:
                    errs.append(f"[9] sơ đồ {n}: {t} không tồn tại ('{line.strip()}')")
                for a in src_ag:
                    if a in reg and t not in reg[a]["tools"]:
                        errs.append(f"[9] sơ đồ {n}: {a} gọi {t} nhưng không được cấp ('{line.strip()}')")
            if src_ag and dst_ag and src != dst and not called and not CHANNEL.search(label):
                errs.append(f"[9] sơ đồ {n}: mũi tên agent→agent thiếu tool/kênh ('{line.strip()}')")
    return errs


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", type=Path, default=PROJECT_ROOT.parent / "docs")
    args = ap.parse_args(argv)
    errs = check(args.docs)
    for e in errs:
        print(f"  ✗ {e}")
    print("[check_docs] OK — truy vết đầy đủ" if not errs else f"[check_docs] {len(errs)} vấn đề")
    return 1 if errs else 0


if __name__ == "__main__":
    sys.exit(main())
