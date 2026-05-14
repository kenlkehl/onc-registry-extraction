"""Download current SEER/NAACCR registry manuals for local prompt retrieval."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import urllib.request
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import quote, unquote, urljoin, urlparse, urlunparse


BASE = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = BASE / "SEERManuals"

SEER_CODING_PAGE = "https://seer.cancer.gov/tools/codingmanuals/index.html"
SEER_APPENDIX_C_PAGE = "https://seer.cancer.gov/manuals/2026/appendixc.html"
SEER_SOLID_TUMOR_PAGE = "https://seer.cancer.gov/tools/solidtumor/"
SEER_EOD_PAGE = "https://seer.cancer.gov/tools/staging/eod/"
SEER_EOD_MANUALS_PAGE = "https://staging.seer.cancer.gov/eod_public/manuals/3.3/"
SEER_SUMMARY_STAGE_PAGE = "https://seer.cancer.gov/tools/ssm/"
NAACCR_SSDI_PAGE = "https://apps.naaccr.org/ssdi/list/"

EXPLICIT_MANUAL_URLS = [
    (
        "staging",
        "EOD 2018 General Instructions",
        "https://seer.cancer.gov/tools/staging/eod/EOD.General%20Instructions.Version3.3.pdf",
    ),
    (
        "staging",
        "EOD Consolidation Manual v3.3",
        "https://seer.cancer.gov/tools/staging/eod/EOD_consolidation_manual_v3.3.pdf",
    ),
    (
        "staging",
        "Summary Stage 2018 Manual v3.3",
        "https://seer.cancer.gov/tools/ssm/Summary-Stage_v3.3.pdf",
    ),
]


class LinkParser(HTMLParser):
    """Small stdlib link extractor."""

    def __init__(self) -> None:
        super().__init__()
        self.links: list[tuple[str, str]] = []
        self._href: str | None = None
        self._text: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "a":
            return
        attrs_dict = {k.lower(): v for k, v in attrs}
        self._href = attrs_dict.get("href")
        self._text = []

    def handle_data(self, data: str) -> None:
        if self._href is not None:
            self._text.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() != "a" or self._href is None:
            return
        text = " ".join(part.strip() for part in self._text if part.strip())
        self.links.append((self._href, re.sub(r"\s+", " ", text).strip()))
        self._href = None
        self._text = []


class TextExtractor(HTMLParser):
    """Lossy HTML to text conversion for source pages."""

    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() in {"br", "p", "li", "tr", "h1", "h2", "h3", "h4"}:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        text = re.sub(r"\s+", " ", data).strip()
        if text:
            self.parts.append(text)

    def text(self) -> str:
        return re.sub(r"\n{3,}", "\n\n", "\n".join(self.parts)).strip()


@dataclass
class ManualEntry:
    category: str
    title: str
    url: str
    local_path: str = ""
    text_path: str = ""
    source_pages: list[str] = field(default_factory=list)


def fetch_bytes(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "onc-registry-pipeline/0.1"})
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


def fetch_text(url: str) -> str:
    return fetch_bytes(url).decode("utf-8", errors="replace")


def parse_links(base_url: str, html: str) -> list[tuple[str, str]]:
    parser = LinkParser()
    parser.feed(html)
    return [(urljoin(base_url, href), text) for href, text in parser.links if href]


def html_to_text(html: str) -> str:
    parser = TextExtractor()
    parser.feed(html)
    return parser.text()


def extension_for_url(url: str) -> str:
    suffix = Path(urlparse(url).path).suffix.lower()
    return suffix or ".html"


def safe_filename(url: str, title: str) -> str:
    parsed = urlparse(url)
    name = Path(parsed.path).name
    if not name:
        name = re.sub(r"[^A-Za-z0-9]+", "_", title).strip("_") or "manual"
        name += extension_for_url(url)
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)


def add_entry(
    entries: dict[str, ManualEntry],
    category: str,
    title: str,
    url: str,
    source_page: str,
) -> None:
    url = normalize_url(url)
    if not url:
        return
    if url in entries:
        if source_page not in entries[url].source_pages:
            entries[url].source_pages.append(source_page)
        return
    entries[url] = ManualEntry(
        category=category,
        title=title or safe_filename(url, ""),
        url=url,
        source_pages=[source_page],
    )


def normalize_url(url: str) -> str:
    """Trim and percent-encode unsafe path characters without changing query."""
    url = url.strip()
    parsed = urlparse(url)
    path = quote(unquote(parsed.path), safe="/:@")
    return urlunparse(parsed._replace(path=path))


def collect_entries() -> tuple[dict[str, ManualEntry], dict[str, tuple[str, str]]]:
    entries: dict[str, ManualEntry] = {}
    source_pages = {
        "seer_coding_manuals": SEER_CODING_PAGE,
        "seer_appendix_c_2026": SEER_APPENDIX_C_PAGE,
        "seer_solid_tumor_rules": SEER_SOLID_TUMOR_PAGE,
        "seer_eod": SEER_EOD_PAGE,
        "seer_eod_manuals": SEER_EOD_MANUALS_PAGE,
        "seer_summary_stage": SEER_SUMMARY_STAGE_PAGE,
        "naaccr_ssdi_grade": NAACCR_SSDI_PAGE,
    }

    html_by_name: dict[str, tuple[str, str]] = {}
    for name, url in source_pages.items():
        html_by_name[name] = (url, fetch_text(url))

    for category, title, url in EXPLICIT_MANUAL_URLS:
        add_entry(entries, category, title, url, "explicit")

    coding_html = html_by_name["seer_coding_manuals"][1]
    for url, title in parse_links(SEER_CODING_PAGE, coding_html):
        path = urlparse(url).path
        if "/manuals/2026/" in path and extension_for_url(url) in {".pdf", ".xlsx"}:
            add_entry(entries, "seer_coding_manual", title, url, "seer_coding_manuals")

    appendix_html = html_by_name["seer_appendix_c_2026"][1]
    for url, title in parse_links(SEER_APPENDIX_C_PAGE, appendix_html):
        path = urlparse(url).path
        if extension_for_url(url) == ".pdf" and (
            "/manuals/2026/AppendixC/" in path
            or "/tools/solidtumor/" in path
            or "/tools/heme/" in path
        ):
            category = "appendix_c"
            if "/tools/solidtumor/" in path:
                category = "solid_tumor_rules"
            elif "/tools/heme/" in path:
                category = "hematopoietic"
            add_entry(entries, category, title, url, "seer_appendix_c_2026")

    solid_html = html_by_name["seer_solid_tumor_rules"][1]
    for url, title in parse_links(SEER_SOLID_TUMOR_PAGE, solid_html):
        if extension_for_url(url) == ".pdf":
            add_entry(entries, "solid_tumor_rules", title, url, "seer_solid_tumor_rules")

    eod_html = html_by_name["seer_eod"][1]
    for url, title in parse_links(SEER_EOD_PAGE, eod_html):
        if extension_for_url(url) == ".pdf":
            add_entry(entries, "staging", title, url, "seer_eod")

    ssdi_html = html_by_name["naaccr_ssdi_grade"][1]
    for url, title in parse_links(NAACCR_SSDI_PAGE, ssdi_html):
        if extension_for_url(url) in {".pdf", ".xlsx"}:
            add_entry(entries, "ssdi_grade", title, url, "naaccr_ssdi_grade")

    return entries, html_by_name


def write_source_pages(
    output_dir: Path, html_by_name: dict[str, tuple[str, str]]
) -> list[ManualEntry]:
    source_dir = output_dir / "source_pages"
    text_dir = output_dir / "text" / "source_pages"
    source_dir.mkdir(parents=True, exist_ok=True)
    text_dir.mkdir(parents=True, exist_ok=True)
    entries = []

    for name, (url, html) in html_by_name.items():
        html_path = source_dir / f"{name}.html"
        text_path = text_dir / f"{name}.txt"
        html_path.write_text(html, encoding="utf-8")
        text_path.write_text(html_to_text(html), encoding="utf-8")
        entries.append(
            ManualEntry(
                category="source_page",
                title=name.replace("_", " ").title(),
                url=url,
                local_path=str(html_path.relative_to(output_dir)),
                text_path=str(text_path.relative_to(output_dir)),
                source_pages=[name],
            )
        )
    return entries


def download_entry(entry: ManualEntry, output_dir: Path) -> None:
    category_dir = output_dir / entry.category
    category_dir.mkdir(parents=True, exist_ok=True)
    filename = safe_filename(entry.url, entry.title)
    local_path = category_dir / filename
    local_path.write_bytes(fetch_bytes(entry.url))
    entry.local_path = str(local_path.relative_to(output_dir))


def extract_text(entry: ManualEntry, output_dir: Path) -> None:
    if not entry.local_path:
        return
    local_path = output_dir / entry.local_path
    suffix = local_path.suffix.lower()
    if suffix == ".pdf":
        text_path = output_dir / "text" / entry.category / f"{local_path.stem}.txt"
        text_path.parent.mkdir(parents=True, exist_ok=True)
        if shutil.which("pdftotext") is None:
            return
        subprocess.run(
            ["pdftotext", "-layout", str(local_path), str(text_path)],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if text_path.exists():
            entry.text_path = str(text_path.relative_to(output_dir))
    elif suffix == ".html":
        text_path = output_dir / "text" / entry.category / f"{local_path.stem}.txt"
        text_path.parent.mkdir(parents=True, exist_ok=True)
        text_path.write_text(html_to_text(local_path.read_text(encoding="utf-8")), encoding="utf-8")
        entry.text_path = str(text_path.relative_to(output_dir))


def write_readme(output_dir: Path, entries: list[ManualEntry]) -> None:
    counts: dict[str, int] = {}
    for entry in entries:
        counts[entry.category] = counts.get(entry.category, 0) + 1
    lines = [
        "# SEER and NAACCR Manuals",
        "",
        "Vendored registry abstraction references used by `onc-registry-pipeline`",
        "to build cancer-type-specific prompt context. Sources are official SEER",
        "and NAACCR pages listed in `manifest.json`.",
        "",
        "## Contents",
        "",
    ]
    for category, count in sorted(counts.items()):
        lines.append(f"- `{category}`: {count} file(s)")
    lines.extend(
        [
            "",
            "PDF text extractions are stored under `text/` for prompt retrieval.",
            "Regenerate this folder with `uv run python scripts/download_seer_manuals.py`.",
            "",
        ]
    )
    (output_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    entries_by_url, html_by_name = collect_entries()
    entries = list(entries_by_url.values())

    for idx, entry in enumerate(entries, start=1):
        print(f"[{idx:03d}/{len(entries):03d}] {entry.title} -> {entry.category}")
        download_entry(entry, output_dir)
        extract_text(entry, output_dir)

    source_entries = write_source_pages(output_dir, html_by_name)
    all_entries = entries + source_entries

    manifest = [
        {
            "category": entry.category,
            "title": entry.title,
            "url": entry.url,
            "local_path": entry.local_path,
            "text_path": entry.text_path,
            "source_pages": entry.source_pages,
        }
        for entry in sorted(all_entries, key=lambda e: (e.category, e.title, e.url))
    ]
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_readme(output_dir, all_entries)

    print(f"Wrote {len(all_entries)} manual/source entries to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
