"""Ảnh minh họa KB từ Neo4j (thay cho ảnh chụp Neo4j Browser) → docs/assets/kb_schema.png, kb_dtd_tip2.png.

Chạy (cần Neo4j):  python -m diabetes_mas.devtools.plot_kb

- kb_schema.png   : meta-graph nhãn → quan hệ → nhãn, kèm số quan hệ (như `CALL db.schema.visualization()`).
- kb_dtd_tip2.png : đồ thị con quanh ĐTĐ típ 2 = truy vấn demo #1 ở 00_KB_GRAPH.md §7 (bỏ MENTIONS), cộng tiêu
                    chuẩn chẩn đoán và biến chứng của nút cha ĐTĐ. Gen, Anatomy và thuốc từ Hetionet gộp thành
                    nút đếm; chỉ vẽ riêng thuốc có nguồn Vinmec.
"""
from __future__ import annotations

import math
import sys
from collections import defaultdict

from diabetes_mas.config import PROJECT_ROOT
from diabetes_mas.kb.neo4j_io import driver

ASSETS = PROJECT_ROOT.parent / "docs" / "assets"
COLORS = {"Disease": "#d1495b", "Symptom": "#edae49", "RiskFactor": "#66a182", "Drug": "#00798c",
          "LabTest": "#8d6cab", "DiagnosticCriterion": "#b07aa1", "Gene": "#9e9e9e", "Anatomy": "#bdbdbd",
          "Source": "#4f6d7a", "Chunk": "#c0d6df"}

SCHEMA_Q = "MATCH (a)-[r]->(b) RETURN labels(a)[0] AS a, type(r) AS rel, labels(b)[0] AS b, count(*) AS n"
LABEL_Q = "MATCH (n) RETURN labels(n)[0] AS l, count(*) AS n"
SUB_Q = """
MATCH (d:Disease {id:'DTD_TIP_2'})-[r]-(x) WHERE type(r) IN ['HAS_SYMPTOM','HAS_RISK_FACTOR','TREATS',
      'DIFFERENTIAL_OF','HAS_COMPLICATION']
RETURN labels(x)[0] AS lab, coalesce(x.name_vi, x.name) AS name, type(r) AS rel, r.source_id AS src
"""
PARENT_Q = """
MATCH (:Disease {id:'DTD'})-[r:DIAGNOSED_BY]->(c)-[:USES_TEST]->(t:LabTest)
RETURN t.name_vi AS name, c.op AS op, c.value AS v, c.unit AS unit, r.source_id AS src, 'crit' AS kind
UNION ALL
MATCH (:Disease {id:'DTD'})-[r:HAS_COMPLICATION]->(x)
RETURN x.name_vi AS name, null AS op, null AS v, null AS unit, r.source_id AS src, 'comp' AS kind
"""
AGG_Q = "MATCH (:Disease {id:'DTD_TIP_2'})-[r:ASSOCIATED_GENE|AFFECTS]-() RETURN type(r) AS rel, count(*) AS n"


def run(q: str) -> list[dict]:
    with driver() as drv, drv.session() as s:
        return s.run(q).data()


def dot(n: int) -> str:
    """3164 → "3.164" (dấu chấm ngăn nghìn kiểu Việt)."""
    return f"{n:,}".replace(",", ".")


def short(s: str, n: int = 26) -> str:
    s = (s or "").split("(")[0].strip()
    return s if len(s) <= n else s[: n - 1] + "…"


def plot_schema(plt) -> None:
    rows = run(SCHEMA_Q)
    counts = {r["l"]: r["n"] for r in run(LABEL_Q)}
    pos = {"Disease": (0, 0), "Symptom": (-2.2, 1.4), "RiskFactor": (-2.4, -0.2), "LabTest": (1.2, -2.2),
           "DiagnosticCriterion": (-0.6, -2.2), "Drug": (2.4, 1.2), "Gene": (2.6, -0.4), "Anatomy": (0.6, 2.2),
           "Chunk": (-0.2, 3.6), "Source": (-2.4, 3.4)}
    fig, ax = plt.subplots(figsize=(13, 9))
    for n, (x, y) in pos.items():
        ax.scatter([x], [y], s=4200, color=COLORS[n], zorder=2, edgecolors="white", linewidths=2)
        label = "Diagnostic\nCriterion" if n == "DiagnosticCriterion" else n
        ax.text(x, y, f"{label}\n{dot(counts.get(n, 0))}", ha="center", va="center", fontsize=9, fontweight="bold",
                color="white" if n not in ("Chunk", "Anatomy") else "#333", zorder=3)
    seen: dict[tuple[str, str], int] = defaultdict(int)
    loops = []
    for r in rows:
        a, b = r["a"], r["b"]
        if a == b:
            loops.append(r)
            continue
        k = seen[(a, b)]
        seen[(a, b)] += 1
        rad = 0.15 + 0.18 * k
        ax.annotate("", xy=pos[b], xytext=pos[a], zorder=1,
                    arrowprops=dict(arrowstyle="-|>", color="#555", lw=1.2, shrinkA=32, shrinkB=32,
                                    connectionstyle=f"arc3,rad={rad}"))
        mx, my = (pos[a][0] + pos[b][0]) / 2, (pos[a][1] + pos[b][1]) / 2
        dx, dy = pos[b][1] - pos[a][1], pos[a][0] - pos[b][0]
        ax.text(mx + 0.5 * rad * dx, my + 0.5 * rad * dy, f"{r['rel']}\n{dot(r['n'])}", fontsize=7.5, ha="center",
                va="center", color="#222", bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#ddd"), zorder=4)
    x, y = pos["Disease"]
    ax.text(x + 0.45, y - 0.55, "Disease → Disease:\n" + "\n".join(f"{r['rel']} {r['n']}" for r in loops),
            fontsize=8, ha="left", va="top", zorder=4,
            bbox=dict(boxstyle="round,pad=0.3", fc="#fdecee", ec=COLORS["Disease"]))
    ax.set_title(f"Schema KB trong Neo4j — {dot(sum(counts.values()))} node · {dot(sum(r['n'] for r in rows))} quan hệ",
                 fontsize=13)
    ax.axis("off")
    ax.set_xlim(-3.4, 3.6)
    ax.set_ylim(-3.0, 4.3)
    fig.tight_layout()
    fig.savefig(ASSETS / "kb_schema.png", dpi=130)
    plt.close(fig)


def plot_subgraph(plt) -> None:
    sub, parent, agg = run(SUB_Q), run(PARENT_Q), {r["rel"]: r["n"] for r in run(AGG_Q)}
    nodes: dict[str, tuple[str, tuple[float, float]]] = {}
    edges: list[tuple[str, str]] = []
    C, P = "ĐTĐ típ 2", "ĐTĐ (nút cha)"
    nodes[C], nodes[P] = ("Disease", (0.0, 0.0)), ("Disease", (0.0, -3.0))

    def ring(names, lab, a0, a1, r, out=True):
        for i, n in enumerate(names):
            a = math.radians(a0 + (a1 - a0) * (i + 0.5) / max(1, len(names)))
            nodes[n] = (lab, (r * math.cos(a), r * math.sin(a)))
            edges.append((C, n) if out else (n, C))

    by: dict[str, list[dict]] = defaultdict(list)
    for r in sub:
        by[r["rel"]].append(r)
    ring([short(r["name"]) for r in by["HAS_SYMPTOM"]], "Symptom", 112, 218, 3.4)
    ring([short(r["name"]) for r in by["HAS_RISK_FACTOR"]], "RiskFactor", -22, 68, 3.4)
    kb_drugs = [short(r["name"]) for r in by["TREATS"] if not str(r["src"]).startswith("HETIONET")]
    n_het = sum(1 for r in by["TREATS"] if str(r["src"]).startswith("HETIONET"))
    ring(kb_drugs + [f"+{n_het} thuốc (Hetionet)"], "Drug", 226, 268, 3.6, out=False)
    for r in by["HAS_COMPLICATION"]:
        nodes[short(r["name"])] = ("Disease", (1.6, -1.7))
        edges.append((C, short(r["name"])))
    for r in by["DIFFERENTIAL_OF"]:
        nodes[short(r["name"])] = ("Disease", (-1.4, -1.6))
        edges.append((short(r["name"]), C))
    edges.append((C, P))
    gene, anat = f"{agg.get('ASSOCIATED_GENE', 0)} gen (Hetionet)", f"{agg.get('AFFECTS', 0)} cơ quan (Hetionet)"
    nodes[gene], nodes[anat] = ("Gene", (1.0, 4.1)), ("Anatomy", (-1.0, 4.1))
    edges += [(C, gene), (C, anat)]
    crits = [r for r in parent if r["kind"] == "crit"]
    comps = [r for r in parent if r["kind"] == "comp"]
    for i, r in enumerate(crits):
        v = f"{r['v']:g}".replace(".", ",")
        n = f"{short(r['name'], 24)} {r['op']} {v} {r['unit'] or ''}\n(nguồn {r['src']})"
        nodes[n] = ("DiagnosticCriterion", (-3.6 + 2.4 * i, -4.6))
        edges.append((P, n))
    for i, r in enumerate(comps):
        n = short(r["name"])
        nodes[n] = ("Disease", (3.6, -2.4 - 0.55 * i))
        edges.append((P, n))

    fig, ax = plt.subplots(figsize=(15, 11))
    for u, v in edges:
        ax.annotate("", xy=nodes[v][1], xytext=nodes[u][1],
                    arrowprops=dict(arrowstyle="-|>", color="#aaa", lw=0.9, shrinkA=22, shrinkB=9))
    for n, (lab, (x, y)) in nodes.items():
        big = n in (C, P)
        ax.scatter([x], [y], s=4200 if big else 300, color=COLORS[lab], zorder=2, edgecolors="white", linewidths=1.5)
        if big:
            ax.text(x, y, n.replace(" (", "\n("), ha="center", va="center", fontsize=10, fontweight="bold",
                    color="white", zorder=3)
        else:
            ax.text(x, y - 0.2, n, ha="center", va="top", fontsize=8, color="#222", zorder=3)
    ax.text(0.12, -2.2, "SUBTYPE_OF", fontsize=8, color="#666")
    handles = [plt.Line2D([], [], marker="o", ls="", color=c, markersize=9, label=l)
               for l, c in COLORS.items() if l not in ("Source", "Chunk", "LabTest")]
    ax.legend(handles=handles, loc="upper left", fontsize=8, frameon=False)
    ax.set_title("KB quanh ĐTĐ típ 2 trong Neo4j (bỏ MENTIONS): triệu chứng, yếu tố nguy cơ, thuốc, biến chứng;\n"
                 "tiêu chuẩn chẩn đoán gắn ở nút cha ĐTĐ, mỗi quan hệ y khoa có source_id", fontsize=12)
    ax.axis("off")
    ax.set_xlim(-5.2, 5.2)
    ax.set_ylim(-5.4, 4.6)
    fig.tight_layout()
    fig.savefig(ASSETS / "kb_dtd_tip2.png", dpi=120)
    plt.close(fig)


def main() -> int:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ASSETS.mkdir(parents=True, exist_ok=True)
    plot_schema(plt)
    plot_subgraph(plt)
    print(f"[plot_kb] → {ASSETS / 'kb_schema.png'}, {ASSETS / 'kb_dtd_tip2.png'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
