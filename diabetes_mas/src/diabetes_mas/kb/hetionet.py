"""Trích phần Hetionet liên quan tới các bệnh seed có `hetionet_id`.

Chỉ lấy: DaG (gen), DlA (giải phẫu), CtD (thuốc điều trị). Bỏ DpS để tránh trùng
triệu chứng tiếng Anh với triệu chứng tiếng Việt đã tuyển chọn.
"""
from __future__ import annotations

import csv
import gzip
from dataclasses import dataclass

from diabetes_mas.config import RAW_DIR

NODES_TSV = RAW_DIR / "hetionet" / "hetionet-v1.0-nodes.tsv"
EDGES_SIF = RAW_DIR / "hetionet" / "hetionet-v1.0-edges.sif.gz"
KEEP = {"DaG": "ASSOCIATED_GENE", "DlA": "AFFECTS", "CtD": "TREATS"}
SOURCE_ID = "HETIONET-1.0-EDGES"


@dataclass
class HetionetSubset:
    genes: list[dict]
    anatomies: list[dict]
    drugs: list[dict]             # thuốc chưa có trong seed
    edges: list[dict]             # rel, from_id, to_id, source_id


def _node_id(het_id: str) -> str:
    kind, ident = het_id.split("::", 1)
    prefix = {"Gene": "GENE", "Anatomy": "ANAT", "Compound": "DRUG"}[kind]
    return f"{prefix}_{ident.replace(':', '_')}"


def extract(disease_map: dict[str, str], drug_name_map: dict[str, str]) -> HetionetSubset:
    """disease_map: hetionet_id -> seed disease id; drug_name_map: tên thuốc (thường) -> seed drug id."""
    names: dict[str, tuple[str, str]] = {}
    with open(NODES_TSV, encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            names[row["id"]] = (row["name"], row["kind"])

    genes, anats, drugs, edges = {}, {}, {}, set()
    with gzip.open(EDGES_SIF, "rt", encoding="utf-8") as f:
        next(f)
        for line in f:
            src, meta, dst = line.rstrip("\n").split("\t")
            if meta not in KEEP:
                continue
            if meta in {"DaG", "DlA"} and src in disease_map:
                name, kind = names[dst]
                nid = _node_id(dst)
                if kind == "Gene":
                    genes[nid] = {"id": nid, "symbol": name, "entrez_id": dst.split("::")[1]}
                else:
                    anats[nid] = {"id": nid, "name_en": name, "uberon_id": dst.split("::")[1]}
                edges.add((KEEP[meta], disease_map[src], nid))
            elif meta == "CtD" and dst in disease_map:
                name, _ = names[src]
                seed_drug = drug_name_map.get(name.lower())
                nid = seed_drug or _node_id(src)
                if not seed_drug:
                    drugs[nid] = {"id": nid, "name": name, "drugbank_id": src.split("::")[1]}
                edges.add(("TREATS", nid, disease_map[dst]))

    return HetionetSubset(
        genes=sorted(genes.values(), key=lambda r: r["id"]),
        anatomies=sorted(anats.values(), key=lambda r: r["id"]),
        drugs=sorted(drugs.values(), key=lambda r: r["id"]),
        edges=[{"rel": r, "from_id": a, "to_id": b, "source_id": SOURCE_ID} for r, a, b in sorted(edges)],
    )
