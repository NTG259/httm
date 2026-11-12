"""Bước 3: dựng đồ thị tri thức tiểu đường từ Hetionet (không ghi tay triple nào).
Cách làm: tìm mọi node Disease có tên chứa "diabet" -> lấy các cạnh nối tới chúng -> lưu CSV."""
import csv
import gzip
from collections import Counter

from config import HETIONET_EDGES, HETIONET_NODES, KEEP_METAEDGES, PROC, RAW


def read_rows(opener, path):
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if parts[0] in ("id", "source"):      # bỏ dòng tiêu đề nếu có
                continue
            yield parts


def main() -> None:
    names, kinds = {}, {}
    for nid, name, kind in read_rows(open, RAW / HETIONET_NODES):
        names[nid], kinds[nid] = name, kind

    diseases = {nid for nid, k in kinds.items()
                if k == "Disease" and "diabet" in names[nid].lower()}
    if not diseases:
        raise SystemExit("Không thấy node Disease nào chứa 'diabet'. Kiểm tra file nodes.")
    print("Các node bệnh được chọn:")
    for nid in sorted(diseases):
        print(f"  {nid}  {names[nid]}")

    rows, counts = [], Counter()
    for s, metaedge, t in read_rows(gzip.open, RAW / HETIONET_EDGES):
        if metaedge in KEEP_METAEDGES and (s in diseases or t in diseases):
            rows.append((names[s], metaedge, names[t], kinds[s], kinds[t], s, t))
            counts[metaedge] += 1

    out = PROC / "diabetes_triples.csv"
    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["subject", "relation", "object", "subject_kind", "object_kind", "subject_id", "object_id"])
        w.writerows(rows)
    print(f"[xong] {len(rows)} triple -> {out}")
    print("Số cạnh theo loại quan hệ:", dict(counts))


if __name__ == "__main__":
    main()
