"""Conservative site-distance and closest-site helpers for grounded chat responses."""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass
from typing import Any, Protocol


LOCATION_DISTANCE_TERMS = (
    "closest",
    "nearest",
    "distance",
    "how far",
    "drive time",
    "driving time",
    "travel time",
    "route",
)


class RoutingProvider(Protocol):
    """Optional drive-time provider abstraction."""

    provider_name: str

    def available(self) -> bool: ...

    def route(
        self,
        *,
        origin: tuple[float, float],
        destination: tuple[float, float],
    ) -> dict[str, Any] | None: ...


class NullRoutingProvider:
    provider_name = "none"

    def available(self) -> bool:
        return False

    def route(
        self,
        *,
        origin: tuple[float, float],
        destination: tuple[float, float],
    ) -> dict[str, Any] | None:
        return None


@dataclass(slots=True)
class SiteCandidate:
    name: str
    contract_name: str | None
    comparison_set_id: str | None
    latitude: float
    longitude: float
    source: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["source"] = self.source
        return payload


@dataclass(slots=True)
class LocationToolDecision:
    query_detected: bool
    supported: bool
    requires_clarification: bool
    routing_tool_available: bool
    drive_time_requested: bool
    drive_time_unavailable: bool
    tool_used: str | None
    route_used: str | None
    answer_text: str | None
    uncertainty_note: str | None
    clarification_reason: str | None
    comparison_scope: list[str]
    origin_label: str | None
    selected_sources: list[dict[str, Any]]
    details: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "query_detected": self.query_detected,
            "supported": self.supported,
            "requires_clarification": self.requires_clarification,
            "routing_tool_available": self.routing_tool_available,
            "drive_time_requested": self.drive_time_requested,
            "drive_time_unavailable": self.drive_time_unavailable,
            "tool_used": self.tool_used,
            "route_used": self.route_used,
            "answer_text": self.answer_text,
            "uncertainty_note": self.uncertainty_note,
            "clarification_reason": self.clarification_reason,
            "comparison_scope": list(self.comparison_scope),
            "origin_label": self.origin_label,
            "selected_source_count": len(self.selected_sources),
            "details": dict(self.details),
        }


def evaluate_location_tool(
    *,
    question: str,
    snippets: list[dict[str, Any]],
    routing_provider: RoutingProvider | None = None,
    max_comparison_sites: int = 5,
) -> LocationToolDecision:
    provider = routing_provider or NullRoutingProvider()
    normalized = _normalize(question)
    query_detected = any(term in normalized for term in LOCATION_DISTANCE_TERMS)
    drive_time_requested = any(
        term in normalized for term in ("drive time", "driving time", "travel time", "route")
    )
    routing_available = provider.available()
    origin = _extract_origin_coordinates(question)
    candidates = _extract_site_candidates(snippets)

    if not query_detected:
        return LocationToolDecision(
            query_detected=False,
            supported=False,
            requires_clarification=False,
            routing_tool_available=routing_available,
            drive_time_requested=drive_time_requested,
            drive_time_unavailable=False,
            tool_used=None,
            route_used=None,
            answer_text=None,
            uncertainty_note=None,
            clarification_reason=None,
            comparison_scope=[],
            origin_label=None,
            selected_sources=[],
            details={"candidate_count": len(candidates)},
        )

    if origin is None:
        return _clarification_decision(
            routing_available=routing_available,
            drive_time_requested=drive_time_requested,
            reason="origin_location_required",
            details={"candidate_count": len(candidates)},
        )

    named_candidates = _named_candidates(normalized, candidates)
    contract_candidates = _contract_scoped_candidates(normalized, candidates, max_comparison_sites)
    comparison_set_candidates = _comparison_set_candidates(candidates, max_comparison_sites)
    origin_label = f"{origin[0]:.4f}, {origin[1]:.4f}"

    if any(term in normalized for term in ("closest", "nearest")):
        scope = _resolve_closest_scope(
            named_candidates=named_candidates,
            contract_candidates=contract_candidates,
            comparison_set_candidates=comparison_set_candidates,
            max_comparison_sites=max_comparison_sites,
        )
        if scope is None:
            return _clarification_decision(
                routing_available=routing_available,
                drive_time_requested=drive_time_requested,
                reason="ambiguous_site_set",
                details={
                    "candidate_count": len(candidates),
                    "named_candidate_count": len(named_candidates),
                    "contract_candidate_count": len(contract_candidates),
                },
            )
        return _distance_decision(
            origin=origin,
            origin_label=origin_label,
            candidates=scope,
            routing_provider=provider,
            drive_time_requested=drive_time_requested,
            comparison_mode=True,
        )

    if len(named_candidates) == 1:
        return _distance_decision(
            origin=origin,
            origin_label=origin_label,
            candidates=named_candidates,
            routing_provider=provider,
            drive_time_requested=drive_time_requested,
            comparison_mode=False,
        )

    if len(candidates) == 1:
        return _distance_decision(
            origin=origin,
            origin_label=origin_label,
            candidates=candidates,
            routing_provider=provider,
            drive_time_requested=drive_time_requested,
            comparison_mode=False,
        )

    return _clarification_decision(
        routing_available=routing_available,
        drive_time_requested=drive_time_requested,
        reason="site_name_required",
        details={
            "candidate_count": len(candidates),
            "named_candidate_count": len(named_candidates),
        },
    )


def _clarification_decision(
    *,
    routing_available: bool,
    drive_time_requested: bool,
    reason: str,
    details: dict[str, Any],
) -> LocationToolDecision:
    clarification_text = (
        "Please name the sites or contracts to compare and provide an origin location. "
        "I only compare a bounded, explicit site set."
    )
    return LocationToolDecision(
        query_detected=True,
        supported=False,
        requires_clarification=True,
        routing_tool_available=routing_available,
        drive_time_requested=drive_time_requested,
        drive_time_unavailable=drive_time_requested and not routing_available,
        tool_used="closest_site_comparison" if "ambiguous_site_set" in reason else "site_distance",
        route_used="clarification_required",
        answer_text=clarification_text,
        uncertainty_note=clarification_text,
        clarification_reason=reason,
        comparison_scope=[],
        origin_label=None,
        selected_sources=[],
        details=details,
    )


def _distance_decision(
    *,
    origin: tuple[float, float],
    origin_label: str,
    candidates: list[SiteCandidate],
    routing_provider: RoutingProvider,
    drive_time_requested: bool,
    comparison_mode: bool,
) -> LocationToolDecision:
    measurements: list[dict[str, Any]] = []
    routing_available = routing_provider.available()
    for candidate in candidates:
        miles = _haversine_miles(origin, (candidate.latitude, candidate.longitude))
        drive_minutes: float | None = None
        if drive_time_requested and routing_available:
            route = routing_provider.route(
                origin=origin,
                destination=(candidate.latitude, candidate.longitude),
            )
            if route:
                drive_minutes = _coerce_float(route.get("drive_minutes"))
        measurements.append(
            {
                "candidate": candidate,
                "distance_miles": miles,
                "drive_minutes": drive_minutes,
            }
        )
    metric_key = "drive_minutes" if drive_time_requested and routing_available else "distance_miles"
    ranked = sorted(
        measurements,
        key=lambda item: (
            item.get(metric_key) if item.get(metric_key) is not None else float("inf"),
            item["distance_miles"],
            item["candidate"].name.casefold(),
        ),
    )
    best = ranked[0]
    best_candidate = best["candidate"]
    drive_time_unavailable = drive_time_requested and not routing_available
    route_used = "drive_time" if drive_time_requested and routing_available else "straight_line_fallback" if drive_time_unavailable else "straight_line"
    tool_used = "closest_site_comparison" if comparison_mode else "site_distance"
    distance_text = f"{best['distance_miles']:.1f} miles"
    if comparison_mode:
        answer = (
            f"Of the explicitly defined comparison set, {best_candidate.name} is the closest site to {origin_label} "
            f"at approximately {distance_text}."
        )
    else:
        answer = f"{best_candidate.name} is approximately {distance_text} from {origin_label}."
    uncertainty_note: str | None = None
    if drive_time_requested and routing_available and best.get("drive_minutes") is not None:
        answer = (
            f"The routed drive time to {best_candidate.name} is about {best['drive_minutes']:.0f} minutes "
            f"from {origin_label}."
        )
        if comparison_mode:
            answer = (
                f"Of the explicitly defined comparison set, {best_candidate.name} has the shortest routed drive time "
                f"at about {best['drive_minutes']:.0f} minutes from {origin_label}."
            )
    elif drive_time_unavailable:
        answer = (
            f"Drive-time routing is unavailable, so this uses straight-line distance only. {answer}"
        )
        uncertainty_note = (
            "Drive-time precision is unavailable without a routing provider, so this result falls back to straight-line distance."
        )
    return LocationToolDecision(
        query_detected=True,
        supported=True,
        requires_clarification=False,
        routing_tool_available=routing_available,
        drive_time_requested=drive_time_requested,
        drive_time_unavailable=drive_time_unavailable,
        tool_used=tool_used,
        route_used=route_used,
        answer_text=answer,
        uncertainty_note=uncertainty_note,
        clarification_reason=None,
        comparison_scope=[item["candidate"].name for item in ranked],
        origin_label=origin_label,
        selected_sources=[item["candidate"].source for item in ranked],
        details={
            "measurements": [
                {
                    "site_name": item["candidate"].name,
                    "contract_name": item["candidate"].contract_name,
                    "distance_miles": round(item["distance_miles"], 3),
                    "drive_minutes": item["drive_minutes"],
                }
                for item in ranked
            ],
            "provider_name": routing_provider.provider_name,
        },
    )


def _resolve_closest_scope(
    *,
    named_candidates: list[SiteCandidate],
    contract_candidates: list[SiteCandidate],
    comparison_set_candidates: list[SiteCandidate],
    max_comparison_sites: int,
) -> list[SiteCandidate] | None:
    if 1 < len(named_candidates) <= max_comparison_sites:
        return named_candidates
    if 1 < len(contract_candidates) <= max_comparison_sites:
        return contract_candidates
    if 1 < len(comparison_set_candidates) <= max_comparison_sites:
        return comparison_set_candidates
    return None


def _named_candidates(normalized_question: str, candidates: list[SiteCandidate]) -> list[SiteCandidate]:
    matches: list[SiteCandidate] = []
    for candidate in candidates:
        name = candidate.name.casefold()
        if name and name in normalized_question:
            matches.append(candidate)
    return _unique_candidates(matches)


def _contract_scoped_candidates(
    normalized_question: str,
    candidates: list[SiteCandidate],
    max_comparison_sites: int,
) -> list[SiteCandidate]:
    grouped: dict[str, list[SiteCandidate]] = {}
    for candidate in candidates:
        if candidate.contract_name:
            grouped.setdefault(candidate.contract_name.casefold(), []).append(candidate)
    for contract_name, scoped in grouped.items():
        if contract_name in normalized_question and 1 < len(scoped) <= max_comparison_sites:
            return _unique_candidates(scoped)
    return []


def _comparison_set_candidates(
    candidates: list[SiteCandidate],
    max_comparison_sites: int,
) -> list[SiteCandidate]:
    grouped: dict[str, list[SiteCandidate]] = {}
    for candidate in candidates:
        if candidate.comparison_set_id:
            grouped.setdefault(candidate.comparison_set_id, []).append(candidate)
    if len(grouped) != 1:
        return []
    scoped = next(iter(grouped.values()))
    if 1 < len(scoped) <= max_comparison_sites:
        return _unique_candidates(scoped)
    return []


def _extract_site_candidates(snippets: list[dict[str, Any]]) -> list[SiteCandidate]:
    candidates: list[SiteCandidate] = []
    for snippet in snippets:
        latitude = _coerce_float(
            snippet.get("latitude")
            or snippet.get("site_latitude")
            or snippet.get("lat")
        )
        longitude = _coerce_float(
            snippet.get("longitude")
            or snippet.get("site_longitude")
            or snippet.get("lon")
            or snippet.get("lng")
        )
        if latitude is None or longitude is None:
            continue
        name = str(
            snippet.get("site_name")
            or snippet.get("site")
            or snippet.get("title")
            or ""
        ).strip()
        if not name:
            continue
        candidates.append(
            SiteCandidate(
                name=name,
                contract_name=str(snippet.get("contract_name") or "").strip() or None,
                comparison_set_id=str(snippet.get("comparison_set_id") or "").strip() or None,
                latitude=latitude,
                longitude=longitude,
                source=snippet,
            )
        )
    return _unique_candidates(candidates)


def _extract_origin_coordinates(question: str) -> tuple[float, float] | None:
    coordinate_pattern = re.compile(
        r"(-?\d{1,3}\.\d+)\s*,\s*(-?\d{1,3}\.\d+)"
    )
    match = coordinate_pattern.search(question)
    if not match:
        return None
    latitude = _coerce_float(match.group(1))
    longitude = _coerce_float(match.group(2))
    if latitude is None or longitude is None:
        return None
    if not (-90 <= latitude <= 90 and -180 <= longitude <= 180):
        return None
    return (latitude, longitude)


def _unique_candidates(candidates: list[SiteCandidate]) -> list[SiteCandidate]:
    deduped: list[SiteCandidate] = []
    seen: set[tuple[str, float, float]] = set()
    for candidate in candidates:
        key = (
            candidate.name.casefold(),
            round(candidate.latitude, 6),
            round(candidate.longitude, 6),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(candidate)
    return deduped


def _normalize(value: str) -> str:
    return re.sub(r"\s+", " ", value.casefold()).strip()


def _coerce_float(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _haversine_miles(
    origin: tuple[float, float],
    destination: tuple[float, float],
) -> float:
    lat1, lon1 = map(math.radians, origin)
    lat2, lon2 = map(math.radians, destination)
    delta_lat = lat2 - lat1
    delta_lon = lon2 - lon1
    a = (
        math.sin(delta_lat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(delta_lon / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return 3958.7613 * c
