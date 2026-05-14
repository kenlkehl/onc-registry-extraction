"""Cancer-type-specific retrieval over vendored SEER/NAACCR manuals."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_WORD_RE = re.compile(r"[a-z0-9]+")
_SITE_CODE_RE = re.compile(r"C(\d{2})", re.IGNORECASE)

_STOP_WORDS = {
    "a",
    "an",
    "and",
    "at",
    "by",
    "cancer",
    "carcinoma",
    "for",
    "from",
    "histology",
    "in",
    "malignant",
    "neoplasm",
    "of",
    "primary",
    "schema",
    "site",
    "the",
    "to",
    "tumor",
    "tumour",
    "unknown",
    "with",
}

_SCHEMA_KEYWORDS: dict[str, list[str]] = {
    "breast": ["breast", "c50"],
    "prostate": ["prostate", "c61", "gleason", "psa"],
    "colon_rectum": ["colon", "rectum", "rectosigmoid", "colorectal", "c18", "c19", "c20", "c21"],
    "lung": ["lung", "bronchus", "c34"],
    "melanoma_skin": ["melanoma", "skin", "c44"],
    "kidney_renal_pelvis": ["kidney", "renal", "pelvis", "ureter", "c64", "c65", "c66"],
    "bladder": ["bladder", "c67"],
    "thyroid": ["thyroid", "c73"],
    "cervix": ["cervix", "cervical", "c53"],
    "ovary": ["ovary", "ovarian", "fallopian", "peritoneal", "c56", "c57"],
    "testis": ["testis", "testicular", "c62"],
    "liver": ["liver", "hepatic", "intrahepatic", "bile", "duct", "c22"],
    "pancreas": ["pancreas", "pancreatic", "c25"],
    "head_neck": [
        "head",
        "neck",
        "tongue",
        "oral",
        "pharynx",
        "larynx",
        "parotid",
        "c00",
        "c01",
        "c02",
        "c03",
        "c04",
        "c05",
        "c06",
        "c07",
        "c08",
        "c09",
        "c10",
        "c11",
        "c12",
        "c13",
        "c14",
        "c30",
        "c31",
        "c32",
    ],
    "brain_cns": ["brain", "cns", "central", "nervous", "intracranial", "c70", "c71", "c72"],
    "generic": [],
}

_HEME_KEYWORDS = {"heme", "hematopoietic", "lymphoma", "leukemia", "myeloma", "lymphoid"}


@dataclass(frozen=True)
class ManualEntry:
    """One vendored manual or extracted source page."""

    category: str
    title: str
    url: str
    local_path: str
    text_path: str
    source_pages: tuple[str, ...]


class SEERManualContextProvider:
    """Build bounded prompt context from local SEER/NAACCR manuals.

    The provider reads `SEERManuals/manifest.json`, scores manuals by
    cancer-type/site/schema keywords, then extracts small local snippets from
    the text-converted PDFs. This keeps prompt context specific to the tumor
    being abstracted and avoids sending entire manuals.
    """

    def __init__(self, manuals_dir: Path, max_chars: int = 12000) -> None:
        self._manuals_dir = manuals_dir
        self._max_chars = max_chars
        self._entries = self._load_manifest()
        self._text_cache: dict[str, str] = {}
        self._context_cache: dict[tuple[str, ...], str] = {}

    @property
    def available(self) -> bool:
        return bool(self._entries)

    def build_context(
        self,
        *,
        tumor_context: str = "",
        cancer_type: str = "",
        primary_site: str = "",
        histology: str = "",
        schema: str = "",
        site_desc: str = "",
        site_context: str = "",
    ) -> str:
        """Return NAACCR/SEER context relevant to the current cancer focus."""
        cache_key = (
            tumor_context,
            cancer_type,
            primary_site,
            histology,
            schema,
            site_desc,
            site_context,
            str(self._max_chars),
        )
        cached = self._context_cache.get(cache_key)
        if cached is not None:
            return cached

        keywords = self._build_keywords(
            tumor_context=tumor_context,
            cancer_type=cancer_type,
            primary_site=primary_site,
            histology=histology,
            schema=schema,
            site_desc=site_desc,
        )

        blocks: list[str] = []
        naaccr_context = self._build_naaccr_context(
            tumor_context=tumor_context,
            cancer_type=cancer_type,
            primary_site=primary_site,
            histology=histology,
            schema=schema,
            site_desc=site_desc,
            site_context=site_context,
        )
        if naaccr_context:
            blocks.append(naaccr_context)

        remaining = max(0, self._max_chars - sum(len(block) for block in blocks) - 400)
        if remaining > 0:
            selected = self._select_entries(keywords)
            snippets = self._build_manual_blocks(selected, keywords, remaining)
            blocks.extend(snippets)

        context = "\n\n".join(blocks).strip()
        if context:
            context = (
                "REGISTRY REFERENCE CONTEXT\n"
                "Use these local NAACCR/SEER references when they clarify coding. "
                "Clinical evidence in the patient text still controls item values.\n\n"
                f"{context}"
            )

        self._context_cache[cache_key] = context
        return context

    def _load_manifest(self) -> list[ManualEntry]:
        manifest_path = self._manuals_dir / "manifest.json"
        if not manifest_path.exists():
            logger.warning("SEER manuals manifest not found: %s", manifest_path)
            return []

        try:
            raw_entries = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            logger.exception("Could not read SEER manuals manifest: %s", manifest_path)
            return []

        entries: list[ManualEntry] = []
        for raw in raw_entries:
            text_path = str(raw.get("text_path", "")).strip()
            if not text_path:
                continue
            if not (self._manuals_dir / text_path).exists():
                continue
            entries.append(
                ManualEntry(
                    category=str(raw.get("category", "")),
                    title=str(raw.get("title", "")),
                    url=str(raw.get("url", "")),
                    local_path=str(raw.get("local_path", "")),
                    text_path=text_path,
                    source_pages=tuple(raw.get("source_pages", [])),
                )
            )
        logger.info("Loaded %d SEER/NAACCR manual text entries", len(entries))
        return entries

    def _build_keywords(
        self,
        *,
        tumor_context: str,
        cancer_type: str,
        primary_site: str,
        histology: str,
        schema: str,
        site_desc: str,
    ) -> list[str]:
        seed_text = " ".join(
            part
            for part in [
                tumor_context,
                cancer_type,
                primary_site,
                histology,
                schema.replace("_", " "),
                site_desc,
            ]
            if part
        )
        keywords = [w for w in _WORD_RE.findall(seed_text.lower()) if w not in _STOP_WORDS]
        keywords.extend(_SCHEMA_KEYWORDS.get(schema, []))

        site_match = _SITE_CODE_RE.search(primary_site)
        if site_match:
            keywords.append(f"c{site_match.group(1)}")

        if any(word in keywords for word in _HEME_KEYWORDS):
            keywords.extend(["hematopoietic", "lymphoid", "heme"])

        seen: set[str] = set()
        deduped: list[str] = []
        for keyword in keywords:
            if len(keyword) < 3 and not keyword.startswith("c"):
                continue
            if keyword not in seen:
                seen.add(keyword)
                deduped.append(keyword)
        return deduped

    def _build_naaccr_context(
        self,
        *,
        tumor_context: str,
        cancer_type: str,
        primary_site: str,
        histology: str,
        schema: str,
        site_desc: str,
        site_context: str,
    ) -> str:
        lines = ["NAACCR cancer-specific context:"]
        if tumor_context:
            lines.append(f"- Tumor focus: {tumor_context}")
        if cancer_type:
            lines.append(f"- Detected diagnosis: {cancer_type}")
        if primary_site:
            lines.append(f"- Resolved primary site: {primary_site}")
        if histology:
            lines.append(f"- Resolved histology: {histology}")
        if schema:
            desc = f" ({site_desc})" if site_desc else ""
            lines.append(f"- Resolved schema: {schema}{desc}")
        if site_context:
            lines.append("Site-specific NAACCR extraction guidance:")
            lines.append(site_context.strip())
        return "\n".join(lines) if len(lines) > 1 else ""

    def _select_entries(self, keywords: list[str]) -> list[ManualEntry]:
        if not keywords:
            return []

        scored: list[tuple[int, ManualEntry]] = []
        for entry in self._entries:
            if entry.category == "source_page":
                continue
            if not entry.text_path:
                continue
            score = self._score_entry(entry, keywords)
            if score > 0:
                scored.append((score, entry))

        scored.sort(
            key=lambda pair: (
                -pair[0],
                self._category_rank(pair[1].category),
                pair[1].title.lower(),
            )
        )
        return [entry for _score, entry in scored[:8]]

    def _score_entry(self, entry: ManualEntry, keywords: list[str]) -> int:
        title_blob = f"{entry.title} {entry.local_path} {entry.category}".lower()
        score = 0
        for keyword in keywords:
            if keyword in title_blob:
                score += 8

        text = self._read_text(entry)
        lower_text = text[:400000].lower()
        for keyword in keywords:
            if keyword in lower_text:
                score += 1

        if entry.category == "appendix_c" and score > 0:
            score += 6
        elif entry.category in {"staging", "ssdi_grade"} and score > 0:
            score += 3
        elif entry.category == "solid_tumor_rules" and score > 0:
            score += 2
        elif entry.category == "hematopoietic" and any(k in _HEME_KEYWORDS for k in keywords):
            score += 10
        return score

    @staticmethod
    def _category_rank(category: str) -> int:
        ranks = {
            "appendix_c": 0,
            "solid_tumor_rules": 1,
            "hematopoietic": 1,
            "staging": 2,
            "ssdi_grade": 3,
            "seer_coding_manual": 4,
        }
        return ranks.get(category, 9)

    def _build_manual_blocks(
        self,
        entries: list[ManualEntry],
        keywords: list[str],
        max_chars: int,
    ) -> list[str]:
        blocks: list[str] = []
        used = 0
        for entry in entries:
            remaining = max_chars - used
            if remaining < 800:
                break
            snippet = self._extract_snippet(
                self._read_text(entry),
                keywords,
                max_chars=min(2400, remaining - 250),
            )
            if not snippet:
                continue
            block = (
                f"SEER/NAACCR manual excerpt: {entry.title}\n"
                f"Source file: {entry.local_path}\n"
                f"{snippet}"
            )
            blocks.append(block)
            used += len(block)
        return blocks

    def _read_text(self, entry: ManualEntry) -> str:
        cached = self._text_cache.get(entry.text_path)
        if cached is not None:
            return cached
        try:
            text = (self._manuals_dir / entry.text_path).read_text(
                encoding="utf-8",
                errors="replace",
            )
        except OSError:
            logger.debug("Could not read manual text: %s", entry.text_path)
            text = ""
        self._text_cache[entry.text_path] = text
        return text

    def _extract_snippet(
        self,
        text: str,
        keywords: list[str],
        max_chars: int,
    ) -> str:
        if not text or max_chars <= 0:
            return ""
        lower_text = text.lower()
        positions: list[int] = []
        for keyword in keywords:
            idx = lower_text.find(keyword)
            if idx >= 0:
                positions.append(idx)
        if not positions:
            return self._clean_text(text[:max_chars])

        windows: list[str] = []
        seen_ranges: list[tuple[int, int]] = []
        window_size = max(800, max_chars // 3)
        for pos in sorted(positions)[:4]:
            start = max(0, pos - window_size // 2)
            end = min(len(text), start + window_size)
            if any(not (end <= s or start >= e) for s, e in seen_ranges):
                continue
            seen_ranges.append((start, end))
            windows.append(self._clean_text(text[start:end]))
            if sum(len(window) for window in windows) >= max_chars:
                break

        snippet = "\n[...]\n".join(window for window in windows if window)
        return snippet[:max_chars].strip()

    @staticmethod
    def _clean_text(text: str) -> str:
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()
