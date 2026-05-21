"""Build prompt messages from canonical prompt and retrieved snippets."""

from __future__ import annotations

import json
from typing import Any

from app.reasoning.intent_router import QueryIntent


def build_ollama_messages(
    *,
    question: str,
    selected_madhhab: str,
    answer_mode: str,
    greeting_style: str,
    tone_level: str,
    retrieved_sources: list[dict[str, Any]],
    prompt_template: str,
    response_schema: dict[str, Any],
    retrieval_policy: dict[str, Any],
    query_intent: QueryIntent | None = None,
    user_context: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    rendered_prompt = prompt_template
    replacements = {
        "{{question}}": question,
        "{{selected_madhhab}}": selected_madhhab,
        "{{answer_mode}}": answer_mode,
        "{{greeting_style}}": greeting_style,
        "{{tone_level}}": tone_level,
        "{{retrieved_sources}}": json.dumps(retrieved_sources, ensure_ascii=False, indent=2),
    }
    for token, value in replacements.items():
        rendered_prompt = rendered_prompt.replace(token, value)

    system_message = "\n\n".join(
        [
            rendered_prompt,
            "Return a single JSON object only.",
            "Do not wrap the response in markdown fences.",
            "Every citation must come from the retrieved sources.",
            (
                "If retrieved sources include source_classification, evidence_bucket, "
                "legal_role, section_label, or quote_window, use them to keep primary "
                "texts, fiqh authorities, spiritual guidance, commentary, and modern rulings clearly distinct."
            ),
            (
                "If non-authoritative user context is provided, use it only for framing, "
                "teaching emphasis, and default presentation. Never let user preference, "
                "project context, or memory override Qur'an, hadith, fiqh manuals, or other retrieved evidence."
            ),
            (
                "If answer_mode is study_path, keep the direct answer concise and educational, "
                "while still respecting that the final study-path layer structure will be assembled from grounded sources."
            ),
            _scholar_perspective_system_instructions(
                answer_mode=answer_mode,
                query_intent=query_intent,
            ),
            (
                "If internal worship_topic is prayer or prayer_method, answer about salat or prayer method directly and "
                "treat wudu or purification only as a prerequisite note when the evidence supports it. "
                "If internal worship_topic is prayer_method, do not answer with a narrow side-topic such as imam posture, "
                "Eid details, or travel prayer unless the retrieved evidence truly centers on that. Prefer step order, pillars, "
                "and the clearest basic prayer anchors. If the retrieved set is partial, say so honestly. "
                "If internal worship_topic is purification or purification_method, focus directly on wudu, ablution, or taharah."
            ),
            (
                "Do not blend Qur'an, hadith, fiqh manuals, tasawwuf texts, commentary, and fatwas "
                "into one undifferentiated claim block."
            ),
            (
                "If tasawwuf texts appear, label them as spiritual guidance and do not present them as legal-ruling authority."
            ),
            (
                "If the policy or intent indicates compare-views, keep distinct positions "
                "separate instead of flattening them."
            ),
            (
                "If the retrieved sources do not justify certainty, use "
                "uncertainty_note and a limited or conflicting evidence_strength."
            ),
            (
                "Internal query intent: "
                f"{query_intent.intent_id if query_intent else 'research'}."
            ),
            (
            "Keep source-only requests citation-heavy and low-synthesis; keep "
            "compare-views requests explicitly distinct instead of blended."
            ),
            (
                "Non-authoritative user context: "
                f"{json.dumps(user_context or {}, ensure_ascii=False)}."
            ),
        ]
    )
    user_message = json.dumps(
        {
            "question": question,
            "selected_madhhab": selected_madhhab,
            "answer_mode": answer_mode,
            "greeting_style": greeting_style,
            "tone_level": tone_level,
            "retrieved_sources": retrieved_sources,
            "query_intent": _query_intent_payload(query_intent),
            "ranking_order_for_legal_questions": retrieval_policy[
                "ranking_order_for_legal_questions"
            ],
            "user_context": user_context or {},
            "output_schema_required_fields": response_schema["required"],
        },
        ensure_ascii=False,
        indent=2,
    )
    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ]


def build_fast_ollama_messages(
    *,
    question: str,
    selected_madhhab: str,
    answer_mode: str,
    greeting_style: str,
    tone_level: str,
    retrieved_sources: list[dict[str, Any]],
    response_schema: dict[str, Any],
    query_intent: QueryIntent | None = None,
    user_context: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    system_message = "\n".join(
        [
            "You are Halal Jordan in fast-path mode.",
            "Return a single JSON object only.",
            "Use only the retrieved sources.",
            "Keep the answer concise, grounded, and citation-first.",
            "Do not invent source layers or certainty.",
            "For source-only requests, avoid synthesis and rely on direct excerpts/citations.",
            "If answer_mode is study_path, keep the direct answer short and overview-like.",
            _scholar_perspective_system_instructions(
                answer_mode=answer_mode,
                query_intent=query_intent,
            ),
            "Treat user preference or project context as non-authoritative framing only.",
            (
                "If internal worship_topic is prayer or prayer_method, answer about prayer first and keep wudu as a prerequisite note only. "
                "If internal worship_topic is prayer_method, avoid turning a general prayer-method question into a narrow subtopic unless the evidence forces that limitation. "
                "If internal worship_topic is purification or purification_method, answer directly about wudu or purification."
            ),
            "If certainty is not supported, use uncertainty_note.",
            f"Internal query intent: {query_intent.intent_id if query_intent else 'research'}.",
        ]
    )
    user_message = json.dumps(
        {
            "question": question,
            "selected_madhhab": selected_madhhab,
            "answer_mode": answer_mode,
            "greeting_style": greeting_style,
            "tone_level": tone_level,
            "response_profile": {
                "path": "fast",
                "style": "concise",
                "avoid_multilayer_explanation": True,
                "max_focus_points": 2,
            },
            "user_context": user_context or {},
            "retrieved_sources": retrieved_sources,
            "query_intent": _query_intent_payload(query_intent),
            "output_schema_required_fields": response_schema["required"],
        },
        ensure_ascii=False,
        indent=2,
    )
    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ]


def _query_intent_payload(query_intent: QueryIntent | None) -> dict[str, Any] | None:
    if query_intent is None:
        return None
    payload: dict[str, Any] = {
        "intent_id": query_intent.intent_id,
        "preserve_disagreement": query_intent.preserve_disagreement,
        "prefer_direct_excerpts": query_intent.prefer_direct_excerpts,
        "prefer_selected_madhhab": query_intent.prefer_selected_madhhab,
        "prefer_primary_texts": query_intent.prefer_primary_texts,
        "prefer_definitional_material": query_intent.prefer_definitional_material,
        "suppress_synthesis": query_intent.suppress_synthesis,
        "target_spiritual_guidance": query_intent.target_spiritual_guidance,
        "target_scholar_commentary": query_intent.target_scholar_commentary,
        "worship_topic": query_intent.worship_topic,
        "authority_order": list(query_intent.authority_order),
        "scholar_id": query_intent.scholar_id,
        "scholar_name": query_intent.scholar_name,
        "scholar_madhhab": query_intent.scholar_madhhab,
        "scholar_period": query_intent.scholar_period,
        "scholar_methodology_notes": query_intent.scholar_methodology_notes,
        "scholar_known_works": list(query_intent.scholar_known_works),
        "scholar_source_families": list(query_intent.scholar_source_families),
        "scholar_retrieval_tags": list(query_intent.scholar_retrieval_tags),
        "unknown_scholar": query_intent.unknown_scholar,
    }
    return payload


def _scholar_perspective_system_instructions(
    *,
    answer_mode: str,
    query_intent: QueryIntent | None,
) -> str:
    active = answer_mode == "scholar_perspective" or (
        query_intent is not None and query_intent.intent_id == "scholar_perspective"
    )
    if not active:
        return "If no scholar-perspective mode is active, do not attribute positions to a named scholar unless the retrieved source text explicitly does so."
    scholar_name = query_intent.scholar_name if query_intent is not None else ""
    label = scholar_name or "the requested scholar"
    return (
        f"If scholar_perspective mode is active, present the answer as drawn from {label}'s recorded works and positions only. "
        "Never invent a ruling or methodology and attribute it to the scholar. "
        "If the retrieved material is thin, explicitly say 'not found in retrieved material from this scholar' or note that attribution is thin. "
        "Keep contextual non-attributed sources secondary and clearly labeled."
    )
