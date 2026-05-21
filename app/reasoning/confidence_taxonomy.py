"""Confidence Taxonomy — scholarly confidence, never numeric.

The charter (and the user's explicit ask) call for scholarly confidence
labels rather than fabricated percentage scores. Numeric confidence in
religious-reasoning UIs implies a precision the system does not have
and the tradition does not claim.

The eight taxonomy levels, from most to least confident:

  1. Explicit text                — supported by clear primary text
  2. Strong consensus             — well-attested ijma claim
  3. Majority position            — jumhur supports
  4. Strong madhhab position      — codified within a madhhab's usul
  5. Valid disagreement           — recognized scholarly difference
  6. Weakly evidenced             — daif support or thin attestation
  7. Speculative                  — opinion without clear textual grounding
  8. Contemporary extrapolation   — modern application of older principles

This module is a rule-based classifier. It reads the evidence model and
assigns ONE label plus a short reasoning sentence. It NEVER outputs a
number. It NEVER inflates: when signals are mixed or weak, the lower
label wins.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

# Ordered most-confident -> least-confident.
LEVEL_ORDER = (
    "explicit_text",
    "strong_consensus",
    "majority_position",
    "strong_madhhab_position",
    "valid_disagreement",
    "weakly_evidenced",
    "speculative",
    "contemporary_extrapolation",
)

LEVEL_LABELS = {
    "explicit_text": "Explicit Text",
    "strong_consensus": "Strong Consensus",
    "majority_position": "Majority Position",
    "strong_madhhab_position": "Strong Madhhab Position",
    "valid_disagreement": "Valid Disagreement",
    "weakly_evidenced": "Weakly Evidenced",
    "speculative": "Speculative",
    "contemporary_extrapolation": "Contemporary Extrapolation",
}

LEVEL_DESCRIPTIONS = {
    "explicit_text": "The answer rests on a direct, clear text from the Qur'an or sahih hadith.",
    "strong_consensus": "A well-attested ijma claim supports the position.",
    "majority_position": "The position is held by the majority (jumhur) of recognized scholars.",
    "strong_madhhab_position": "The position is codified within a madhhab's usul; cross-madhhab divergence may exist.",
    "valid_disagreement": "Recognized scholarly disagreement; the answer represents one position among legitimate alternatives.",
    "weakly_evidenced": "Support rests on weak or limited evidence; treat with caution.",
    "speculative": "An opinion without clear textual grounding; presented as inference.",
    "contemporary_extrapolation": "A modern application of older principles; principle is classical, application is recent.",
}


@dataclass(slots=True, frozen=True)
class ConfidenceAssessment:
    level_id: str
    label: str
    description: str
    reasoning: str  # short explanation of which signals led to this label

    def to_dict(self) -> dict[str, Any]:
        return {
            "level_id": self.level_id,
            "label": self.label,
            "description": self.description,
            "reasoning": self.reasoning,
        }


def classify_confidence(
    *,
    evidence_model: Any | None = None,
    answer: Any | None = None,
) -> ConfidenceAssessment | None:
    """Assign a scholarly confidence label based on observable signals.

    Returns ``None`` when no evidence is available — the renderer should
    stay silent rather than print a fabricated confidence band. This is
    the charter rule (transparency over confidence) enforced at the
    confidence layer.

    Signal hierarchy (highest priority first):
      - explicit ijma claim in citations -> strong_consensus
      - clear primary text in citations + no disagreement -> explicit_text
      - disagreement present (comparison_positions OR disagreement_note)
          -> valid_disagreement
      - daif / weak evidence dominant -> weakly_evidenced
      - majority of sources are modern fatwas -> contemporary_extrapolation
      - madhhab authority sources present + selected madhhab matches
          -> strong_madhhab_position
      - otherwise: speculative
    """

    sources = _collect_sources(evidence_model)
    if not sources:
        return None

    has_explicit_disagreement = _has_disagreement(evidence_model, answer)
    classifications = [_source_classification(s) for s in sources]
    grades = [_hadith_grade(s) for s in sources]
    is_ijma_claimed = _any_ijma_claim(sources, answer)

    # 1. Strong consensus
    if is_ijma_claimed:
        return _assess(
            "strong_consensus",
            reasoning="Cited material includes an attested ijma / consensus claim.",
        )

    primary_text_count = sum(1 for c in classifications if c in {"quran", "hadith"})
    sahih_count = sum(1 for g in grades if g in {"sahih", "mutawatir"})
    daif_count = sum(1 for g in grades if g in {"daif", "mawdu"})
    fiqh_manual_count = sum(1 for c in classifications if c == "fiqh_manual")
    fatwa_count = sum(1 for c in classifications if c == "fatwa")

    # 2. Explicit text (clear primary support, no recognized disagreement,
    # and not undermined by weak gradings). Daif hadith never qualify the
    # answer as resting on explicit text — even if they outnumber other
    # sources, they downgrade to weakly_evidenced below.
    daif_dominates = daif_count and daif_count >= max(1, len(sources) // 2)
    if (
        primary_text_count >= 1
        and not has_explicit_disagreement
        and sahih_count >= 1
        and not daif_dominates
    ):
        return _assess(
            "explicit_text",
            reasoning=(
                f"Supported by {sahih_count} sahih/mutawatir primary text(s) "
                f"with no recognized disagreement."
            ),
        )
    if (
        primary_text_count >= 1
        and not has_explicit_disagreement
        and primary_text_count == len(sources)
        and not daif_dominates
        and daif_count == 0
    ):
        return _assess(
            "explicit_text",
            reasoning="All cited material is primary text with no recognized disagreement.",
        )

    # 3. Valid disagreement (charter rule: preserve, never flatten)
    if has_explicit_disagreement:
        return _assess(
            "valid_disagreement",
            reasoning=(
                "Recognized scholarly disagreement is documented. The answer "
                "represents one position among legitimate alternatives."
            ),
        )

    # 4. Weakly evidenced
    if daif_count and daif_count >= max(1, len(sources) // 2):
        return _assess(
            "weakly_evidenced",
            reasoning=(
                f"{daif_count} of {len(sources)} cited hadith carry weak / "
                f"fabricated gradings."
            ),
        )

    # 5. Contemporary extrapolation
    if fatwa_count and fatwa_count == len(sources):
        return _assess(
            "contemporary_extrapolation",
            reasoning="Only modern fatwa material was cited; no classical authority anchors the answer.",
        )
    if fatwa_count >= max(1, len(sources) * 2 // 3) and primary_text_count == 0:
        return _assess(
            "contemporary_extrapolation",
            reasoning="Cited material is dominated by modern fatwa with no primary-text anchor.",
        )

    # 6. Strong madhhab position
    selected_madhhab = _selected_madhhab(answer, evidence_model)
    madhhab_match_count = sum(
        1
        for s in sources
        if _source_classification(s) in {"fiqh_manual", "commentary"}
        and _madhhab(s) == selected_madhhab
        and selected_madhhab not in {"", "not_specified", "compare_all"}
    )
    if fiqh_manual_count >= 1 and madhhab_match_count >= 1:
        return _assess(
            "strong_madhhab_position",
            reasoning=(
                f"Codified within the {selected_madhhab} school; "
                f"{madhhab_match_count} matching authority source(s) cited."
            ),
        )

    # 7. Majority position (jumhur)
    if fiqh_manual_count >= 2:
        return _assess(
            "majority_position",
            reasoning=(
                f"{fiqh_manual_count} fiqh-manual authorities cited; pattern "
                f"suggests jumhur agreement."
            ),
        )

    # 8. Speculative (default low-confidence floor — never silent inflation)
    return _assess(
        "speculative",
        reasoning=(
            "No primary text, ijma, or codified madhhab authority dominates the "
            "evidence set. Treat as inference."
        ),
    )


def _assess(level_id: str, *, reasoning: str) -> ConfidenceAssessment:
    return ConfidenceAssessment(
        level_id=level_id,
        label=LEVEL_LABELS[level_id],
        description=LEVEL_DESCRIPTIONS[level_id],
        reasoning=reasoning,
    )


# ---------------------------------------------------------------------------
# Signal extraction
# ---------------------------------------------------------------------------


def _collect_sources(evidence_model: Any | None) -> list[Any]:
    if evidence_model is None:
        return []
    sources = getattr(evidence_model, "sources", None)
    if sources:
        return list(sources)
    if isinstance(evidence_model, dict):
        return list(evidence_model.get("sources") or evidence_model.get("citations") or [])
    return []


def _has_disagreement(evidence_model: Any | None, answer: Any | None) -> bool:
    if evidence_model is not None:
        positions = getattr(evidence_model, "comparison_positions", None) or []
        if positions:
            return True
        notes = getattr(evidence_model, "disagreement_notes", None) or []
        if notes:
            return True
    if isinstance(answer, dict):
        note = str(answer.get("disagreement_note") or "").strip()
        if note:
            return True
        mapping = answer.get("disagreement_map")
        if isinstance(mapping, dict) and mapping.get("positions"):
            return True
    return False


def _source_classification(source: Any) -> str:
    value = (
        _attr_or_item(source, "source_classification")
        or _attr_or_item(source, "source_type")
        or ""
    )
    return str(value or "").strip().lower()


def _hadith_grade(source: Any) -> str:
    for key in ("hadith_grade", "authenticity_grade", "authenticity", "authority_level"):
        value = _attr_or_item(source, key)
        if value:
            lowered = str(value).strip().lower()
            if lowered in {"sahih", "saheeh", "authentic"}:
                return "sahih"
            if lowered == "mutawatir":
                return "mutawatir"
            if lowered in {"hasan", "good"}:
                return "hasan"
            if lowered in {"daif", "da'if", "weak"}:
                return "daif"
            if lowered in {"mawdu", "mawdoo", "fabricated"}:
                return "mawdu"
    return ""


def _any_ijma_claim(sources: Iterable[Any], answer: Any | None) -> bool:
    for source in sources:
        for key in ("ijma_strength", "consensus_level", "ijma"):
            value = _attr_or_item(source, key)
            if value and str(value).strip().lower() in {"ijma", "ijmaa", "consensus"}:
                return True
    if isinstance(answer, dict):
        summary = str(answer.get("evidence_summary") or "").lower()
        if "ijma" in summary or "consensus of the scholars" in summary:
            return True
    return False


def _madhhab(source: Any) -> str:
    value = _attr_or_item(source, "madhhab") or ""
    return str(value or "").strip().lower()


def _selected_madhhab(answer: Any | None, evidence_model: Any | None) -> str:
    if isinstance(answer, dict):
        value = str(answer.get("selected_madhhab") or "").strip().lower()
        if value:
            return value
    return ""


def _attr_or_item(source: Any, key: str) -> Any:
    if source is None:
        return None
    if isinstance(source, dict):
        return source.get(key)
    return getattr(source, key, None)


def render_confidence_lines(assessment: ConfidenceAssessment | None) -> list[str]:
    """Render the assessment as plain-text lines. Returns [] if no assessment."""

    if assessment is None:
        return []
    return [
        "Scholarly Confidence",
        f"  Level: {assessment.label}",
        f"  Why: {assessment.reasoning}",
    ]
