"""Dry-run metadata connector for SeekersGuidance Answers."""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse
from urllib.request import Request, urlopen

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

BASE_URL = "https://seekersguidance.org/answers/"
CATEGORY_PREFIX = "https://seekersguidance.org/category/answers/"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "raw" / "fatwas" / "seekersguidance" / "index"
DEFAULT_PROFILE_PATH = REPO_ROOT / "metadata" / "source_registry" / "providers" / "seekersguidance_answers.json"
DEFAULT_CATEGORY_PATH = REPO_ROOT / "metadata" / "taxonomies" / "seekersguidance_categories.json"
SNIPPET_LIMIT = 320


@dataclass(slots=True)
class SeekersGuidanceRecord:
    source_name: str
    provider: str
    source_type: str
    source_role_boundary: str
    authority_layer: str
    title: str
    url: str
    scholar: str
    date: str
    reading_time: str
    category: str
    categories: list[str]
    madhhab_tag: str
    summary_snippet: str
    admission_status: str
    copyright_status: str
    storage_mode: str
    provenance_note: str
    fetched_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_name": self.source_name,
            "provider": self.provider,
            "source_type": self.source_type,
            "source_role_boundary": self.source_role_boundary,
            "authority_layer": self.authority_layer,
            "title": self.title,
            "url": self.url,
            "scholar": self.scholar,
            "date": self.date,
            "reading_time": self.reading_time,
            "category": self.category,
            "categories": self.categories,
            "madhhab_tag": self.madhhab_tag,
            "summary_snippet": self.summary_snippet,
            "admission_status": self.admission_status,
            "copyright_status": self.copyright_status,
            "storage_mode": self.storage_mode,
            "provenance_note": self.provenance_note,
            "fetched_at": self.fetched_at,
        }


class _CategoryAnchorParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._capture = False
        self._current_href = ""
        self._parts: list[str] = []
        self.categories: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "a":
            return
        attributes = dict(attrs)
        href = str(attributes.get("href") or "")
        if href.startswith(CATEGORY_PREFIX):
            self._capture = True
            self._current_href = href
            self._parts = []

    def handle_data(self, data: str) -> None:
        if self._capture:
            self._parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag != "a" or not self._capture:
            return
        label = _normalize_space("".join(self._parts))
        if label and label not in self.categories:
            self.categories.append(label)
        self._capture = False
        self._current_href = ""
        self._parts = []


def load_seekersguidance_profile(path: Path = DEFAULT_PROFILE_PATH) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_seekersguidance_categories(path: Path = DEFAULT_CATEGORY_PATH) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def extract_categories_from_landing(html: str) -> dict[str, Any]:
    parser = _CategoryAnchorParser()
    parser.feed(html)
    categories = parser.categories
    return {
        "source_name": "SeekersGuidance Answers",
        "all_categories": categories,
        "topic_categories": [label for label in categories if not _is_school_specific_category(label)],
        "madhhab_specific_categories": {
            "hanafi": [label for label in categories if _category_targets_madhhab(label) == "hanafi"],
            "shafii": [label for label in categories if _category_targets_madhhab(label) == "shafii"],
            "maliki": [label for label in categories if _category_targets_madhhab(label) == "maliki"],
            "hanbali": [label for label in categories if _category_targets_madhhab(label) == "hanbali"],
        },
        "worship_categories": [label for label in categories if _is_worship_category(label)],
        "transactions_family_categories": [
            label for label in categories if _is_transactions_family_category(label)
        ],
    }


def fetch_answer_metadata(url: str, *, fetcher: Callable[[str], str] | None = None) -> SeekersGuidanceRecord:
    normalized_url = _normalize_answer_url(url)
    html = (fetcher or fetch_url_html)(normalized_url)
    profile = load_seekersguidance_profile()
    categories = _extract_answer_categories(html)
    title = _extract_title(html)
    scholar = _extract_scholar(html)
    published = _extract_published_date(html)
    reading_time = _extract_reading_time(html)
    snippet = _extract_summary_snippet(html)
    madhhab_tag = _infer_madhhab_from_categories(categories)
    return SeekersGuidanceRecord(
        source_name=str(profile["source_name"]),
        provider=str(profile["provider"]),
        source_type=str(profile["source_type"]),
        source_role_boundary=str(profile["source_role_boundary"]),
        authority_layer=str(profile["authority_layer"]),
        title=title,
        url=normalized_url,
        scholar=scholar,
        date=published,
        reading_time=reading_time,
        category=categories[0] if categories else "",
        categories=categories,
        madhhab_tag=madhhab_tag,
        summary_snippet=snippet,
        admission_status=str(profile["default_admission_status"]),
        copyright_status=str(profile["copyright_status"]),
        storage_mode="metadata_snippet_only",
        provenance_note="External reference only. Metadata and short snippet captured from allowlisted URL.",
        fetched_at=_utc_now(),
    )


def intake_allowlist(
    urls: list[str],
    *,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    fetcher: Callable[[str], str] | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for url in urls:
        record = fetch_answer_metadata(url, fetcher=fetcher)
        record_dict = record.to_dict()
        target_path = output_dir / f"{_slug_from_url(record.url)}.json"
        target_path.write_text(
            json.dumps(record_dict, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        records.append(record_dict)
    index_path = output_dir / "index.json"
    index_path.write_text(
        json.dumps(
            {
                "source_name": "SeekersGuidance Answers",
                "storage_mode": "metadata_snippet_only",
                "record_count": len(records),
                "records": [
                    {
                        "title": record["title"],
                        "url": record["url"],
                        "scholar": record["scholar"],
                        "category": record["category"],
                        "madhhab_tag": record["madhhab_tag"],
                        "admission_status": record["admission_status"],
                        "copyright_status": record["copyright_status"],
                    }
                    for record in records
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "output_dir": str(output_dir),
        "record_count": len(records),
        "records": records,
        "index_path": str(index_path),
    }


def fetch_url_html(url: str) -> str:
    request = Request(
        url,
        headers={
            "User-Agent": "HalalJordan/1.0 metadata-only dry-run connector",
        },
    )
    with urlopen(request, timeout=15) as response:
        return response.read().decode("utf-8", errors="replace")


def _normalize_answer_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("SeekersGuidance URL must use http or https.")
    if parsed.netloc not in {"seekersguidance.org", "www.seekersguidance.org"}:
        raise ValueError("Only seekersguidance.org URLs are allowed.")
    if parsed.query:
        raise ValueError("Query-string URLs are not allowed for SeekersGuidance intake.")
    normalized_path = parsed.path.rstrip("/") + "/"
    if not normalized_path.startswith("/answers/") or normalized_path == "/answers/":
        raise ValueError("Only allowlisted answer detail pages under /answers/ are allowed.")
    return f"https://seekersguidance.org{normalized_path}"


def _extract_title(html: str) -> str:
    return (
        _extract_meta(html, "og:title")
        or _strip_tags(_first_match(html, r"<h1[^>]*>(.*?)</h1>", flags=re.IGNORECASE | re.DOTALL))
        or "SeekersGuidance Answer"
    )


def _extract_scholar(html: str) -> str:
    scholar = _strip_tags(
        _first_match(
            html,
            r"Answered by(?:&nbsp;|\s|<[^>]+>)*([^<]+)",
            flags=re.IGNORECASE | re.DOTALL,
        )
    )
    return scholar or "Unknown Scholar"


def _extract_published_date(html: str) -> str:
    published = _extract_meta(html, "article:published_time")
    if published:
        return published
    return _strip_tags(
        _first_match(
            html,
            r"\[\s*([A-Za-z]+\s+\d{1,2},\s+\d{4})\s*\]",
            flags=re.IGNORECASE,
        )
    )


def _extract_reading_time(html: str) -> str:
    return _normalize_space(
        _first_match(
            html,
            r"(\d+\s+mins?)",
            flags=re.IGNORECASE,
        )
    )


def _extract_summary_snippet(html: str) -> str:
    snippet = _extract_meta(html, "og:description") or _extract_meta(html, "description")
    if not snippet:
        answer_section = _first_match(
            html,
            r"<h3>\s*Answer\s*</h3>(.*?)(?:<h[34][^>]*>|<footer[^>]*>)",
            flags=re.IGNORECASE | re.DOTALL,
        )
        paragraphs = re.findall(r"<p[^>]*>(.*?)</p>", answer_section, flags=re.IGNORECASE | re.DOTALL)
        cleaned = [_strip_tags(paragraph) for paragraph in paragraphs]
        snippet = " ".join(part for part in cleaned[:2] if part)
    snippet = _normalize_space(snippet)
    if len(snippet) > SNIPPET_LIMIT:
        snippet = snippet[: SNIPPET_LIMIT - 3].rsplit(" ", 1)[0] + "..."
    return snippet


def _extract_answer_categories(html: str) -> list[str]:
    raw_categories = re.findall(
        r'href="https://seekersguidance\.org/category/answers/[^"]+/"[^>]*>\s*([^<]+?)\s*</a>',
        html,
        flags=re.IGNORECASE,
    )
    categories: list[str] = []
    for item in raw_categories:
        label = _normalize_space(unescape(item))
        if label and label not in categories:
            categories.append(label)
    return categories


def _infer_madhhab_from_categories(categories: list[str]) -> str:
    for category in categories:
        candidate = _category_targets_madhhab(category)
        if candidate:
            return candidate
    return ""


def _category_targets_madhhab(label: str) -> str:
    lowered = label.casefold()
    if "hanafi" in lowered:
        return "hanafi"
    if "shafii" in lowered or "shafi'i" in lowered:
        return "shafii"
    if "maliki" in lowered:
        return "maliki"
    if "hanbali" in lowered:
        return "hanbali"
    return ""


def _is_school_specific_category(label: str) -> bool:
    return bool(_category_targets_madhhab(label))


def _is_worship_category(label: str) -> bool:
    lowered = label.casefold()
    return any(
        token in lowered
        for token in (
            "fast",
            "hajj and umra",
            "prayer",
            "purity",
            "ramadan",
            "remembrance",
            "sacrifice",
            "supplication",
            "zakat",
        )
    )


def _is_transactions_family_category(label: str) -> bool:
    lowered = label.casefold()
    return any(
        token in lowered
        for token in (
            "birth",
            "children",
            "family",
            "inheritance",
            "intimacy",
            "jobs and income",
            "marriage and divorce",
            "oaths",
            "parents",
            "transactions",
        )
    )


def _extract_meta(html: str, key: str) -> str:
    pattern = re.compile(
        rf'<meta[^>]+(?:property|name)="(?:{re.escape(key)})"[^>]+content="([^"]*)"',
        flags=re.IGNORECASE,
    )
    match = pattern.search(html)
    return _normalize_space(unescape(match.group(1))) if match else ""


def _first_match(text: str, pattern: str, *, flags: int = 0) -> str:
    match = re.search(pattern, text, flags=flags)
    return match.group(1) if match else ""


def _strip_tags(value: str) -> str:
    if not value:
        return ""
    no_tags = re.sub(r"<[^>]+>", " ", value)
    return _normalize_space(unescape(no_tags))


def _normalize_space(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _slug_from_url(url: str) -> str:
    path = urlparse(url).path.strip("/")
    slug = path.rsplit("/", 1)[-1]
    slug = re.sub(r"[^a-z0-9-]+", "-", slug.casefold()).strip("-")
    return slug or "seekersguidance-answer"


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Dry-run metadata-only connector for SeekersGuidance Answers."
    )
    parser.add_argument("--url", action="append", default=[], help="Allowlisted answer URL to fetch.")
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for metadata-only output records.",
    )
    return parser


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = _build_arg_parser().parse_args()
    if not args.url:
        raise SystemExit("Provide at least one --url for dry-run intake.")
    result = intake_allowlist(args.url, output_dir=Path(args.output_dir))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
