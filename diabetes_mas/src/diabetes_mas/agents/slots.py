"""Kho slot: hợp nhất slot y khoa từ KB seed + slot hồ sơ từ configs/slots.yaml.

Chạy:  python -m diabetes_mas.agents.slots     → in kho slot, không gian hành động từng worker, d_s
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from diabetes_mas.config import load_yaml
from diabetes_mas.kb.seed import load_seed

DISEASES = ["KHONG_DTD", "TIEN_DTD", "DTD_TIP_1", "DTD_TIP_2", "DTD_THAI_KY"]   # 5 lớp in_scope (thứ tự cố định)
WORKERS = ["W1", "W2", "W3", "W4"]


@dataclass(frozen=True)
class Slot:
    id: str
    kind: str                     # binary | numeric | categorical
    source: str                   # Symptom | RiskFactor | LabTest | profile | derived
    group: str | None
    worker: str | None            # Work-Agent được hỏi slot này; None = không hỏi trực tiếp
    question_vi: str
    values: tuple[str, ...] = ()  # với categorical
    range: tuple[float, float] | None = None


@dataclass(frozen=True)
class SlotCatalog:
    slots: tuple[Slot, ...]

    def by_id(self) -> dict[str, Slot]:
        return {s.id: s for s in self.slots}

    def actions(self, worker: str) -> list[str]:
        return [s.id for s in self.slots if s.worker == worker]

    def askable(self) -> list[str]:
        return [s.id for s in self.slots if s.worker]


@lru_cache(maxsize=1)
def catalog() -> SlotCatalog:
    cfg = load_yaml("slots.yaml")
    seed = load_seed()
    derived = cfg["derived"]
    not_askable = set(cfg["not_askable_groups"])
    group_to_worker = {g: w for w, spec in cfg["workers"].items() for g in spec.get("kb_groups", [])}
    lab_worker = next(w for w, spec in cfg["workers"].items() if "LabTest" in spec.get("kb_labels", []))
    profile_worker = next(w for w, spec in cfg["workers"].items() if spec.get("profile"))
    ranges = cfg["numeric_ranges"]

    slots: list[Slot] = []
    for s in seed.nodes["Symptom"]:
        worker = None if s["group"] in not_askable else group_to_worker.get(s["group"])
        slots.append(Slot(s["id"], "binary", "Symptom", s["group"], worker, s.get("question_vi", "")))
    for r in seed.nodes["RiskFactor"]:
        is_derived = r["id"] in derived
        slots.append(Slot(r["id"], "binary", "derived" if is_derived else "RiskFactor", r["group"],
                          None if is_derived else group_to_worker.get(r["group"]), r.get("question_vi", "")))
    for t in seed.nodes["LabTest"]:
        rng = tuple(ranges[t["id"]]) if t["id"] in ranges else None
        slots.append(Slot(t["id"], t["kind"], "LabTest", "xet_nghiem", lab_worker, t.get("question_vi", ""), range=rng))
    for pid, spec in cfg["profile"].items():
        slots.append(Slot(pid, spec["kind"], "profile", "ho_so", profile_worker, spec["question_vi"],
                          values=tuple(spec.get("values", [])),
                          range=tuple(spec["range"]) if "range" in spec else None))
    for s in slots:
        if s.kind == "numeric" and s.range is None:
            raise ValueError(f"Slot số {s.id} thiếu numeric_ranges trong slots.yaml")
    return SlotCatalog(tuple(sorted(slots, key=lambda x: (x.source, x.id))))


def derive_flags(values: dict[str, float]) -> dict[str, bool]:
    """Suy ra cờ nhị phân (tuoi_tu_45, …) từ slot số đã biết."""
    out = {}
    for flag, rule in load_yaml("slots.yaml")["derived"].items():
        x = values.get(rule["from"])
        if x is None:
            continue
        out[flag] = x >= rule["value"] if rule["op"] == ">=" else x < rule["value"]
    return out


def main() -> None:
    from diabetes_mas.agents.state import state_dim

    cat = catalog()
    print(f"{'slot':<22}{'kind':<12}{'nguồn':<11}{'worker':<7}")
    for s in cat.slots:
        print(f"{s.id:<22}{s.kind:<12}{s.source:<11}{s.worker or '-':<7}")
    for w in WORKERS:
        acts = cat.actions(w)
        print(f"{w}: {len(acts)} slot + RETURN → |A| = {len(acts) + 1}")
    print(f"Main Agent: |A| = {len(WORKERS) + 1} (W1..W4, CLASSIFY)")
    print(f"d_s = {state_dim()}")


if __name__ == "__main__":
    main()
