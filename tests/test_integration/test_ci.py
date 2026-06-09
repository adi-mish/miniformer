from pathlib import Path


def test_github_actions_ci_runs_verification_gate():
    workflow = Path(__file__).resolve().parents[2] / ".github" / "workflows" / "ci.yml"
    text = workflow.read_text()

    assert 'python-version: ["3.10", "3.11", "3.12"]' in text
    assert "uv sync --extra dev --extra docs" in text
    assert "uv run miniformer-check" in text
