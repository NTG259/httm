"""Thu thập nguồn tĩnh khai báo trong configs/sources.yaml (PDF Bộ Y tế, WHO, Hetionet).

Chạy:  python -m diabetes_mas.ingest.fetch_static

Ưu tiên `local_file` (sao chép, không di chuyển); nếu không có thì tải từ `url`.
Ghi manifest data/raw/static_manifest.jsonl (id, đường dẫn, sha256, nguồn gốc).
"""
from __future__ import annotations

import hashlib
import shutil
import sys
from datetime import datetime, timezone

from diabetes_mas.config import PROJECT_ROOT, RAW_DIR, load_yaml
from diabetes_mas.ingest.http import PoliteClient
from diabetes_mas.utils.text import write_jsonl

MANIFEST = RAW_DIR / "static_manifest.jsonl"


def _sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    entries = load_yaml("sources.yaml")["static"]
    client: PoliteClient | None = None
    manifest = []
    for e in entries:
        dest = RAW_DIR / e["dest"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        origin = None
        if not dest.exists():
            local = (PROJECT_ROOT / e["local_file"]).resolve() if e.get("local_file") else None
            if local and local.exists():
                shutil.copy2(local, dest)
                origin = f"copied:{local}"
            elif e.get("url"):
                client = client or PoliteClient()
                resp = client.get(e["url"])
                resp.raise_for_status()
                dest.write_bytes(resp.content)
                origin = f"downloaded:{e['url']}"
            else:
                print(f"[thiếu] {e['id']}: không có local_file hợp lệ hay url", file=sys.stderr)
                continue
            print(f"[ok] {e['id']} ← {origin}")
        manifest.append({
            "id": e["id"], "file": e["dest"], "sha256": _sha256_file(dest),
            "origin": origin or "existing", "checked_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        })
    write_jsonl(MANIFEST, manifest)
    print(f"[manifest] {len(manifest)} nguồn → {MANIFEST}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
