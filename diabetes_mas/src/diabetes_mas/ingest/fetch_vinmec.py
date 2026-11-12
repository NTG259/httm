"""Thu thập bài Vinmec qua sitemap (SRC-VMC).

Chạy:  python -m diabetes_mas.ingest.fetch_vinmec [--max-posts N] [--diseases-only]

- Trang bệnh (/vie/benh/...) được tải trước.
- Bài viết (/vie/bai-viet/...) lọc theo slug trong configs/settings.yaml.
- Có thể chạy lại: bỏ qua file đã tải (theo URL trong index.jsonl).
- Trang chuyển hướng tới /khong-tim-thay hoặc HTTP != 200 bị bỏ qua.
"""
from __future__ import annotations

import argparse
import re
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone

from diabetes_mas.config import RAW_DIR, settings
from diabetes_mas.ingest.http import PoliteClient, RobotsDisallowed
from diabetes_mas.utils.text import append_jsonl, read_jsonl, sha256_text

OUT_DIR = RAW_DIR / "vinmec"
INDEX = OUT_DIR / "index.jsonl"
SITEMAP_NS = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
SLUG_RE = re.compile(r"/vie/(?:bai-viet|benh)/([^/?#]+)")


def _locs(xml_text: str) -> list[str]:
    root = ET.fromstring(xml_text)
    return [el.text.strip() for el in root.iter(f"{{{SITEMAP_NS['sm']}}}loc") if el.text]


def discover_posts(client: PoliteClient, keywords: list[str]) -> list[str]:
    index_url = settings()["vinmec"]["sitemap_index"]
    sitemaps = [u for u in _locs(client.get(index_url).text) if "/posts-vi-" in u]
    print(f"[sitemap] {len(sitemaps)} sitemap bài viết tiếng Việt", flush=True)
    found: list[str] = []
    for i, sm in enumerate(sitemaps, 1):
        try:
            urls = _locs(client.get(sm).text)
        except Exception as exc:  # noqa: BLE001 — 1 sitemap lỗi không dừng cả lượt
            print(f"[sitemap] lỗi {sm}: {exc}", flush=True)
            continue
        hits = [u for u in urls if any(k in u.lower() for k in keywords)]
        found.extend(hits)
        if i % 10 == 0:
            print(f"[sitemap] {i}/{len(sitemaps)} — {len(found)} URL khớp", flush=True)
    return sorted(set(found))


def _next_source_id(existing: list[dict]) -> int:
    nums = [int(r["source_id"].split("-")[1]) for r in existing if r.get("source_id", "").startswith("VMC-")]
    return max(nums, default=0) + 1


def fetch(urls: list[str], client: PoliteClient, kind: str) -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    existing = list(read_jsonl(INDEX))
    done = {r["url"] for r in existing}
    seen_hash = {r["sha256"] for r in existing}
    next_id = _next_source_id(existing)
    n_new = 0
    for url in urls:
        if url in done:
            continue
        try:
            resp = client.get(url)
        except RobotsDisallowed:
            print(f"[skip robots] {url}", flush=True)
            continue
        except Exception as exc:  # noqa: BLE001
            print(f"[lỗi] {url}: {exc}", flush=True)
            continue
        if resp.status_code != 200 or "khong-tim-thay" in resp.url:
            print(f"[skip {resp.status_code}] {url}", flush=True)
            continue
        html = resp.text
        digest = sha256_text(html)
        if digest in seen_hash:
            continue
        sid = f"VMC-{next_id:04d}"
        next_id += 1
        fname = f"{sid}.html"
        (OUT_DIR / fname).write_text(html, encoding="utf-8")
        m = SLUG_RE.search(url)
        append_jsonl(INDEX, {
            "source_id": sid,
            "url": url,
            "kind": kind,
            "slug": m.group(1) if m else None,
            "file": f"vinmec/{fname}",
            "sha256": digest,
            "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        })
        seen_hash.add(digest)
        n_new += 1
        print(f"[ok] {sid} {url}", flush=True)
    return n_new


def main(argv: list[str] | None = None) -> int:
    cfg = settings()["vinmec"]
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-posts", type=int, default=cfg["max_posts"])
    ap.add_argument("--diseases-only", action="store_true")
    args = ap.parse_args(argv)

    client = PoliteClient()
    robots_snapshot = OUT_DIR / f"_robots_{datetime.now():%Y-%m-%d}.txt"
    if not robots_snapshot.exists():
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        robots_snapshot.write_text(client.get("https://www.vinmec.com/robots.txt").text, encoding="utf-8")

    n = fetch(cfg["disease_pages"], client, kind="disease")
    print(f"[benh] tải mới {n} trang", flush=True)
    if args.diseases_only:
        return 0

    posts = discover_posts(client, cfg["post_slug_keywords"])
    print(f"[bai-viet] {len(posts)} URL khớp từ khóa; tải tối đa {args.max_posts}", flush=True)
    n = fetch(posts[: args.max_posts], client, kind="post")
    print(f"[bai-viet] tải mới {n} bài", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
