"""Tách nội dung chính từ HTML Vinmec thành các mục (heading + text).

Hai dạng trang:
- Bài viết (/vie/bai-viet/...): nội dung trong #main-article, mục chia theo h2/h3.
- Trang bệnh (/vie/benh/...): mỗi mục là div.body.collapsible-target, tiêu đề là h2 đứng trước.

Văn bản lấy theo **khối lá** (p, li, h*, td…) bằng get_text("") để không bẻ đôi từ
ở các thẻ inline như <strong> (vd. "tiền sản giật").
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from bs4 import BeautifulSoup, Tag

from diabetes_mas.utils.text import collapse_ws

BLOCK_TAGS = {"h2", "h3", "h4", "p", "li", "td", "th", "blockquote", "dt", "dd"}
HEADING_TAGS = {"h2", "h3", "h4"}
DROP_SELECTORS = ["script", "style", "figure", "noscript", "iframe", "section.table-of-contents"]
CTA_PATTERNS = re.compile(
    r"(để đặt lịch khám|hotline|my vinmec|đặt lịch trực tiếp|tải và đặt lịch|bấm số|đăng ký khám)", re.I
)
TITLE_SUFFIX = re.compile(r"\s*\|\s*Vinmec.*$")


@dataclass
class Section:
    heading: str
    text: str


@dataclass
class ParsedPage:
    title: str
    published: str | None
    modified: str | None
    description: str | None
    author: str | None
    sections: list[Section] = field(default_factory=list)


def _jsonld_article(soup: BeautifulSoup) -> dict:
    for s in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(s.string or "")
        except (json.JSONDecodeError, TypeError):
            continue
        items = data.get("@graph", [data]) if isinstance(data, dict) else data
        for it in items:
            if isinstance(it, dict) and it.get("@type") in {"NewsArticle", "Article", "MedicalWebPage", "BlogPosting"}:
                return it
    return {}


def _is_leaf_block(el: Tag) -> bool:
    return el.name in BLOCK_TAGS and el.find(BLOCK_TAGS) is None


def _leaf_blocks(root: Tag) -> list[Tag]:
    return [el for el in root.find_all(BLOCK_TAGS) if _is_leaf_block(el)]


def _clean_block(el: Tag) -> str:
    return collapse_ws(el.get_text(""))


def _sections_from_blocks(blocks: list[Tag], default_heading: str) -> list[Section]:
    sections: list[Section] = []
    heading, buf = default_heading, []
    seen: set[str] = set()

    def flush() -> None:
        text = "\n".join(buf).strip()
        if text:
            sections.append(Section(heading=heading, text=text))

    for el in blocks:
        txt = _clean_block(el)
        if not txt or txt in {"☰", "Mục lục"}:
            continue
        if el.name in HEADING_TAGS:
            flush()
            heading, buf = txt, []
            continue
        if CTA_PATTERNS.search(txt):
            continue
        if txt in seen:          # Vinmec lặp nội dung li (mobile/desktop)
            continue
        seen.add(txt)
        if el.name == "li":
            txt = f"- {txt}"
        buf.append(txt)
    flush()
    return sections


def _strip(root: Tag) -> None:
    for sel in DROP_SELECTORS:
        for el in root.select(sel):
            el.decompose()
    for el in root.find_all("section"):
        if el.find(string=re.compile("Mục lục")):
            el.decompose()


def parse_post(html: str) -> ParsedPage:
    soup = BeautifulSoup(html, "html.parser")
    meta = _jsonld_article(soup)
    h1 = soup.find("h1")
    title = meta.get("headline") or (h1.get_text(" ", strip=True) if h1 else TITLE_SUFFIX.sub("", soup.title.string or ""))
    root = soup.select_one("#main-article")
    sections: list[Section] = []
    if root is not None:
        _strip(root)
        sections = _sections_from_blocks(_leaf_blocks(root), default_heading=title)
    author = meta.get("author", {})
    return ParsedPage(
        title=collapse_ws(title),
        published=meta.get("datePublished"),
        modified=meta.get("dateModified"),
        description=collapse_ws(meta.get("description", "")) or None,
        author=author.get("name") if isinstance(author, dict) else None,
        sections=sections,
    )


def parse_disease(html: str) -> ParsedPage:
    soup = BeautifulSoup(html, "html.parser")
    meta = _jsonld_article(soup)
    raw_title = soup.title.string if soup.title else ""
    title = collapse_ws(TITLE_SUFFIX.sub("", raw_title))
    sections: list[Section] = []
    for body in soup.select("div.body.collapsible-target"):
        prev = body.find_previous(["h2", "h3"])
        heading = collapse_ws(prev.get_text("")) if prev else title
        _strip(body)
        sub = _sections_from_blocks(_leaf_blocks(body), default_heading=heading)
        # gộp các mục con (h3 bên trong) vào mục cha để giữ ngữ cảnh "Triệu chứng bệnh X"
        for s in sub:
            h = heading if s.heading == heading else f"{heading} › {s.heading}"
            sections.append(Section(heading=h, text=s.text))
    desc = soup.find("meta", attrs={"name": "description"})
    return ParsedPage(
        title=title,
        published=meta.get("datePublished"),
        modified=meta.get("dateModified"),
        description=collapse_ws(desc["content"]) if desc and desc.get("content") else None,
        author=None,
        sections=sections,
    )


def parse(html: str, kind: str) -> ParsedPage:
    return parse_disease(html) if kind == "disease" else parse_post(html)
