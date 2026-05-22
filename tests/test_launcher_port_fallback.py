"""Tests for the launcher's port-fallback logic.

We can't load PowerShell from Python pytest, but we CAN verify the
fallback logic is present in the launcher script as the user-facing
contract: a busy preferred port no longer aborts; it tries alternates
and announces the chosen one.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER_PATH = REPO_ROOT / "LAUNCH_HALAL_JORDAN.ps1"


def _launcher_text() -> str:
    return LAUNCHER_PATH.read_text(encoding="utf-8", errors="replace")


def test_resolve_launch_port_function_exists():
    text = _launcher_text()
    assert "function Resolve-LaunchPort" in text


def test_port_fallback_loop_is_present():
    """The fallback iterates a range of ports, not just one — that's
    what removes the 'stop the other app' friction.
    """

    text = _launcher_text()
    # The loop signature; brittle but anchors the contract
    assert "for (" in text and "FallbackRange" in text


def test_port_fallback_default_range_is_at_least_5():
    """Five alternates is the minimum useful range — fewer and a
    single conflicting dev environment kills the launch.
    """

    text = _launcher_text()
    # Find the default value of FallbackRange parameter
    import re
    match = re.search(r"\[int\]\$FallbackRange\s*=\s*(\d+)", text)
    assert match is not None, "Resolve-LaunchPort must declare $FallbackRange with a default"
    assert int(match.group(1)) >= 5, "fallback range should cover at least 5 alternate ports"


def test_port_fallback_logs_chosen_alternate():
    """When the launcher falls back to a different port, it must
    announce the new port so the user knows where to open the browser.
    """

    text = _launcher_text()
    assert "fallback port" in text.lower()


def test_port_fallback_only_throws_when_range_exhausted():
    """The launcher must not throw on the first conflict — only after
    every port in the configured range is taken.
    """

    text = _launcher_text()
    # The throw is gated by exhausting the loop, not by a single hit
    # Sanity: look for 'no free port in range' in the error message
    assert "no free port in range" in text.lower()
