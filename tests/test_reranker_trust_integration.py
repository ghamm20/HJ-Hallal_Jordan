"""Integration tests for trust_engine wiring inside the reranker.

These guard the two invariants that the wiring must preserve:

1. Default profile = zero behavioral change. Order returned by
   rerank_candidates with trust_profile_id="default" must equal the order
   returned by an identical call without trust_profile_id at all.
2. Non-default profile actually changes ordering when signals are present.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.retrieval.reranker import rerank_candidates  # noqa: E402


RETRIEVAL_POLICY = {
    "ranking_order_for_legal_questions": [
        "quran",
        "hadith",
        "fiqh_manual",
        "commentary",
        "fatwa",
        "tasawwuf_text",
        "scholar_transcript",
        "transcript",
    ]
}
SOURCE_TYPE_REGISTRY = {
    "source_types": [
        {"id": "quran", "display_priority": 0},
        {"id": "hadith", "display_priority": 1},
        {"id": "fiqh_manual", "display_priority": 2},
        {"id": "commentary", "display_priority": 3},
        {"id": "fatwa", "display_priority": 4},
        {"id": "tasawwuf_text", "display_priority": 5},
        {"id": "scholar_transcript", "display_priority": 6},
        {"id": "transcript", "display_priority": 7},
    ]
}


def _make_candidate(**overrides):
    base = {
        "title": overrides.get("title", "Untitled"),
        "source_type": overrides.get("source_type", "fiqh_manual"),
        "source_classification": overrides.get("source_type", "fiqh_manual"),
        "madhhab": overrides.get("madhhab", ""),
        "retrieval_score": overrides.get("retrieval_score", 0.5),
        "reference": overrides.get("reference", "ref"),
        "source_path": overrides.get("source_path", "x.pdf"),
        "text": overrides.get("text", "body"),
    }
    base.update(overrides)
    return base


def _ids(results):
    return [c["title"] for c in results]


def test_default_profile_preserves_baseline_ordering():
    candidates = [
        _make_candidate(title="hanafi_manual", madhhab="hanafi", retrieval_score=0.5),
        _make_candidate(title="shafii_manual", madhhab="shafii", retrieval_score=0.55),
        _make_candidate(title="hanbali_manual", madhhab="hanbali", retrieval_score=0.45),
    ]

    baseline = rerank_candidates(
        [dict(c) for c in candidates],
        selected_madhhab="not_specified",
        retrieval_policy=RETRIEVAL_POLICY,
        source_type_registry=SOURCE_TYPE_REGISTRY,
        top_k=3,
    )
    with_default = rerank_candidates(
        [dict(c) for c in candidates],
        selected_madhhab="not_specified",
        retrieval_policy=RETRIEVAL_POLICY,
        source_type_registry=SOURCE_TYPE_REGISTRY,
        top_k=3,
        trust_profile_id="default",
    )

    assert _ids(baseline) == _ids(with_default), (
        "Default trust profile must not change reranker ordering"
    )


def test_hanafi_heavy_profile_lifts_hanafi_above_higher_scored_shafii():
    """With a small retrieval_score gap and a Hanafi-heavy trust profile,
    the Hanafi candidate should rise above the slightly-higher-scored
    Shafi'i one.
    """

    candidates = [
        _make_candidate(
            title="hanafi_manual",
            madhhab="hanafi",
            source_type="fiqh_manual",
            era="classical",
            retrieval_score=0.50,
        ),
        _make_candidate(
            title="shafii_manual",
            madhhab="shafii",
            source_type="fiqh_manual",
            era="classical",
            retrieval_score=0.55,
        ),
    ]

    ranked = rerank_candidates(
        candidates,
        selected_madhhab="not_specified",
        retrieval_policy=RETRIEVAL_POLICY,
        source_type_registry=SOURCE_TYPE_REGISTRY,
        top_k=2,
        trust_profile_id="hanafi_heavy",
    )
    assert _ids(ranked)[0] == "hanafi_manual"


def test_breakdown_attached_to_every_candidate():
    candidates = [_make_candidate(title="a"), _make_candidate(title="b")]
    ranked = rerank_candidates(
        candidates,
        selected_madhhab="not_specified",
        retrieval_policy=RETRIEVAL_POLICY,
        source_type_registry=SOURCE_TYPE_REGISTRY,
        top_k=2,
        trust_profile_id="hadith_focused",
    )
    for candidate in ranked:
        assert "_trust_breakdown" in candidate
        assert candidate["_trust_breakdown"]["profile_id"] == "hadith_focused"
