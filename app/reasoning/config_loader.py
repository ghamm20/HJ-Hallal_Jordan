"""Load canonical contract artifacts for the ask pipeline."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class ContractArtifacts:
    voice_profiles: dict[str, Any]
    answer_modes: dict[str, Any]
    source_type_registry: dict[str, Any]
    retrieval_policy: dict[str, Any]
    answer_response_schema: dict[str, Any]
    prompt_template: str
    default_greeting_style: str
    default_tone_level: str
    greeting_style_ids: set[str]
    tone_level_ids: set[str]
    answer_mode_ids: set[str]
    source_type_ids: set[str]
    selected_madhhab_ids: set[str]


def load_contract_artifacts(repo_root: Path) -> ContractArtifacts:
    metadata = repo_root / "metadata"
    voice_profiles = _load_json(metadata / "taxonomies" / "voice_profiles.json")
    answer_modes = _load_json(metadata / "taxonomies" / "answer_modes.json")
    source_type_registry = _load_json(
        metadata / "source_registry" / "source_type_registry.json"
    )
    retrieval_policy = _load_json(metadata / "mappings" / "retrieval_policy.json")
    answer_response_schema = _load_json(
        metadata / "schemas" / "answer_response.schema.json"
    )
    prompt_template = (
        repo_root / "prompts" / "retrieval_grounded_answer.md"
    ).read_text(encoding="utf-8")

    return ContractArtifacts(
        voice_profiles=voice_profiles,
        answer_modes=answer_modes,
        source_type_registry=source_type_registry,
        retrieval_policy=retrieval_policy,
        answer_response_schema=answer_response_schema,
        prompt_template=prompt_template,
        default_greeting_style=voice_profiles["default_profile"]["greeting_style"],
        default_tone_level=voice_profiles["default_profile"]["tone_level"],
        greeting_style_ids={item["id"] for item in voice_profiles["greeting_styles"]},
        tone_level_ids={item["id"] for item in voice_profiles["tone_levels"]},
        answer_mode_ids={item["id"] for item in answer_modes["modes"]},
        source_type_ids={item["id"] for item in source_type_registry["source_types"]},
        selected_madhhab_ids=set(
            answer_response_schema["properties"]["selected_madhhab"]["enum"]
        ),
    )


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))
