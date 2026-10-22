"""Guard that the headline numbers quoted in the README still match
reports/results.json, so the two cannot silently drift. Skipped if results
have not been generated (e.g. a fresh clone before evaluation)."""

import json
from pathlib import Path

import pytest

README = Path("README.md")
RESULTS = Path("reports/results.json")


@pytest.mark.skipif(not RESULTS.is_file(), reason="results.json not generated yet")
def test_readme_quotes_match_results() -> None:
    results = json.loads(RESULTS.read_text(encoding="utf-8"))
    metrics = results["metrics"]
    text = README.read_text(encoding="utf-8")

    for raw in (
        metrics["escape_rate"]["raw"],
        metrics["defect_recall"]["raw"],
        metrics["false_reject_rate"]["raw"],
    ):
        assert raw in text, f"README is missing the raw count {raw} from results.json"


def test_readme_has_no_video_or_demo_script_section() -> None:
    text = README.read_text(encoding="utf-8").lower()
    # The README must not reference a recorded demo video (project decision).
    assert "youtube" not in text
    assert "demo video" not in text
    assert "watch the demo" not in text


def test_readme_has_ownership_note() -> None:
    text = README.read_text(encoding="utf-8").lower()
    assert "personal project" in text
    assert "no affiliation" in text
