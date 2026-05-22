"""End-to-end flight test of the live Halal Jordan server.

Pings each endpoint a real user would touch and captures the output to
flight_test_results.txt. Run with the server already listening on
127.0.0.1:8000.
"""

from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request

BASE = "http://127.0.0.1:8000"
OUT = []


def out(line: str = "") -> None:
    OUT.append(line)
    print(line, flush=True)


def wait_for_ready(timeout: int = 600) -> bool:
    out(f"Waiting for server at {BASE} (up to {timeout}s)...")
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = urllib.request.urlopen(BASE + "/health", timeout=3)
            if r.status == 200:
                out(f"  Ready in {int(time.time() - (deadline - timeout))}s")
                return True
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            time.sleep(1)
    out("  TIMEOUT")
    return False


def http(method: str, path: str, body: dict | None = None) -> tuple[int, str]:
    data = None
    headers = {}
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(BASE + path, data=data, headers=headers, method=method)
    try:
        r = urllib.request.urlopen(req, timeout=300)
        return r.status, r.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", errors="replace")


def section(title: str) -> None:
    out("")
    out("=" * 70)
    out(title)
    out("=" * 70)


def main() -> int:
    if not wait_for_ready():
        return 1

    section("STEP 1 — GET /  (should now be the ask page)")
    status, body = http("GET", "/")
    out(f"  status: {status}")
    out(f"  contains 'Your question': {'Your question' in body}")
    out(f"  contains '/api/ask':      {'/api/ask' in body}")
    out(f"  contains 'Clear conversation': {'Clear conversation' in body}")
    out(f"  body length: {len(body)} chars")

    section("STEP 2 — GET /workspace  (full chat UI preserved)")
    status, body = http("GET", "/workspace")
    out(f"  status: {status}")
    out(f"  body length: {len(body)} chars (workspace is larger)")

    section("STEP 3 — GET /profiles  (button selector)")
    status, body = http("GET", "/profiles")
    out(f"  status: {status}")
    out(f"  contains 'Scholar Methodology': {'Scholar Methodology' in body}")
    out(f"  contains 'Shaykh Jamal':        {'Shaykh Jamal' in body}")
    out(f"  contains 'Dr. Umar':            {'Dr. Umar' in body}")
    out(f"  contains 'Methodology modeling': {'Methodology modeling' in body}")

    section("STEP 4 — GET /api/profile/list")
    status, body = http("GET", "/api/profile/list")
    payload = json.loads(body)
    out(f"  status: {status}")
    out(f"  profile count: {len(payload['profiles'])}")
    for p in payload["profiles"]:
        flag = "[SCHOLAR]" if p["is_scholar_methodology"] else "         "
        out(f"    {flag} {p['profile_id']:<30s} mode={p['mode']}")

    section("STEP 5 — POST /api/profile/set hadith_focused")
    status, body = http("POST", "/api/profile/set", {"profile_id": "hadith_focused"})
    out(f"  status: {status}")
    out(f"  body: {body}")

    section("STEP 6 — POST /api/ask  (first question, hadith_focused profile)")
    status, body = http(
        "POST",
        "/api/ask",
        {"question": "intentions sincerity", "selected_madhhab": "not_specified"},
    )
    out(f"  status: {status}")
    if status == 200:
        data = json.loads(body)
        out(f"  profile_id:     {data['profile_id']}")
        out(f"  source_count:   {data['source_count']}")
        out(f"  confidence:     {data['confidence']['label'] if data.get('confidence') else '(none)'}")
        out(f"  ladder tiers:")
        for tier in data["evidence_ladder"]:
            out(f"    Tier {tier['rank']} {tier['label']}: {tier['count']} source(s)")
        out("")
        out("  --- rendered_text (first 40 lines) ---")
        ascii_text = data["rendered_text"].encode("ascii", "replace").decode("ascii")
        for line in ascii_text.split("\n")[:40]:
            out(f"    {line}")
        first_question = data["question"]
    else:
        out(f"  ERROR body: {body[:500]}")
        first_question = None

    section("STEP 7 — POST /api/ask  (follow-up with conversation_context)")
    if first_question:
        status, body = http(
            "POST",
            "/api/ask",
            {
                "question": "what about prayer?",
                "selected_madhhab": "not_specified",
                "conversation_context": [first_question],
            },
        )
        out(f"  status: {status}")
        if status == 200:
            data = json.loads(body)
            out(f"  retrieval_question used: {data['retrieval_question'][:120]!r}")
            out(f"  conversation_context_used: {data['conversation_context_used']}")
            out(f"  source_count:   {data['source_count']}")
            out(f"  confidence:     {data['confidence']['label'] if data.get('confidence') else '(none)'}")
        else:
            out(f"  ERROR body: {body[:500]}")

    section("STEP 8 — POST /api/profile/set back to default")
    status, body = http("POST", "/api/profile/set", {"profile_id": "default"})
    out(f"  status: {status} body: {body}")

    section("STEP 9 — POST /api/ask with empty question (should 400)")
    status, body = http("POST", "/api/ask", {"question": ""})
    out(f"  status: {status} (expected 400)  body: {body[:200]}")

    section("STEP 10 — POST /api/profile/set unknown (should 404)")
    status, body = http("POST", "/api/profile/set", {"profile_id": "not_a_real_profile"})
    out(f"  status: {status} (expected 404)  body: {body[:200]}")

    section("DONE")
    return 0


if __name__ == "__main__":
    rc = main()
    with open("flight_test_results.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(OUT))
    sys.exit(rc)
