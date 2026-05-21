"""Disagreement Mapping — show *where* divergence occurred and *why*.

The charter requires that the system never flatten real differences
across madhhabs or source classes, and that uncertainty be shown rather
than hidden. The user's explicit ask sharpens this: not just "scholars
differed" but a structural map of:

  - where the divergence occurred
  - which principle of usul caused it
  - what evidence each side prioritized

This module defines the data structures and a defensive parser that
takes optional input from the answer JSON and produces a clean
``DisagreementMap``. The renderer surfaces whatever structured data is
present and stays silent otherwise. The code never fabricates
disagreement — if the model didn't provide structured divergence data,
the renderer falls back to whatever free-text ``disagreement_note`` the
existing pipeline carries.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping


@dataclass(slots=True, frozen=True)
class DisagreementPosition:
    label: str  # e.g., "Hanafi" or "Jumhur"
    holders: tuple[str, ...] = ()
    evidence_priorities: tuple[str, ...] = ()
    ruling_summary: str = ""
    citations: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "holders": list(self.holders),
            "evidence_priorities": list(self.evidence_priorities),
            "ruling_summary": self.ruling_summary,
            "citations": list(self.citations),
        }


@dataclass(slots=True, frozen=True)
class DisagreementMap:
    point: str  # what the disagreement is about
    principle: str  # which usul principle caused divergence
    positions: tuple[DisagreementPosition, ...] = ()
    notes: tuple[str, ...] = ()

    def is_empty(self) -> bool:
        return not (self.point.strip() or self.principle.strip() or self.positions)

    def to_dict(self) -> dict[str, Any]:
        return {
            "point": self.point,
            "principle": self.principle,
            "positions": [p.to_dict() for p in self.positions],
            "notes": list(self.notes),
        }


def parse_disagreement_map(raw: Any) -> DisagreementMap | None:
    """Defensive parse of optional model-provided disagreement data.

    Accepts None, an empty container, a dict, or a list of position
    dicts. Returns ``None`` when there is nothing structured to show —
    the renderer falls back to the existing free-text disagreement
    note. Never invents fields.
    """

    if not raw:
        return None
    if isinstance(raw, list):
        positions = tuple(_parse_position(item) for item in raw if _is_mapping(item))
        positions = tuple(p for p in positions if p)
        if not positions:
            return None
        return DisagreementMap(point="", principle="", positions=positions)
    if not _is_mapping(raw):
        return None
    point = str(raw.get("point") or "").strip()
    principle = str(raw.get("principle") or "").strip()
    raw_positions = raw.get("positions") or []
    positions = tuple(
        _parse_position(item) for item in raw_positions if _is_mapping(item)
    )
    positions = tuple(p for p in positions if p)
    notes = tuple(
        str(note).strip()
        for note in (raw.get("notes") or [])
        if isinstance(note, str) and note.strip()
    )
    if not (point or principle or positions or notes):
        return None
    return DisagreementMap(
        point=point,
        principle=principle,
        positions=positions,
        notes=notes,
    )


def _parse_position(raw: Mapping[str, Any]) -> DisagreementPosition | None:
    label = str(raw.get("label") or "").strip()
    if not label:
        return None
    holders = _str_tuple(raw.get("holders"))
    evidence_priorities = _str_tuple(raw.get("evidence_priorities"))
    ruling_summary = str(raw.get("ruling_summary") or raw.get("summary") or "").strip()
    citations = _str_tuple(raw.get("citations") or raw.get("references"))
    return DisagreementPosition(
        label=label,
        holders=holders,
        evidence_priorities=evidence_priorities,
        ruling_summary=ruling_summary,
        citations=citations,
    )


def _str_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        items = [chunk.strip() for chunk in value.split(",")]
        return tuple(item for item in items if item)
    if isinstance(value, Iterable):
        return tuple(
            str(item).strip()
            for item in value
            if str(item).strip()
        )
    return ()


def _is_mapping(value: Any) -> bool:
    return isinstance(value, Mapping)


def render_disagreement_lines(disagreement: DisagreementMap | None) -> list[str]:
    """Render the disagreement map as plain-text lines.

    Returns ``[]`` when no structured map is present so the renderer
    stays silent rather than printing an empty section. The existing
    free-text ``disagreement_note`` is still rendered separately by the
    answer pipeline; this is the structural surface on top.
    """

    if disagreement is None or disagreement.is_empty():
        return []
    lines = ["Where Scholars Diverged"]
    if disagreement.point:
        lines.append(f"  Point: {disagreement.point}")
    if disagreement.principle:
        lines.append(f"  Principle of divergence: {disagreement.principle}")
    for position in disagreement.positions:
        lines.append(f"  - {position.label}")
        if position.holders:
            lines.append(f"      Held by: {', '.join(position.holders)}")
        if position.ruling_summary:
            lines.append(f"      Ruling: {position.ruling_summary}")
        if position.evidence_priorities:
            lines.append(
                "      Evidence prioritized: "
                + "; ".join(position.evidence_priorities)
            )
        if position.citations:
            for citation in position.citations:
                lines.append(f"      Cites: {citation}")
    for note in disagreement.notes:
        lines.append(f"  Note: {note}")
    return lines
