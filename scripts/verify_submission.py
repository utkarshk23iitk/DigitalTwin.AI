"""Fail fast when submission-only files are missing or generated assets are tracked."""

from __future__ import annotations

import py_compile
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
REQUIRED_FILES = (
    "app2.py",
    "README.md",
    "requirements.txt",
    "docs/SUBMISSION_REPORT.md",
    "data/generate_training_data.py",
    "data/generate_demo_data.py",
    "data/build_demo_inference.py",
    "src/line_sim.py",
    "src/virtual_sensor.py",
    "src/defect_model.py",
    "src/effective_trust.py",
)
ALLOWED_GENERATED_PLACEHOLDERS = {
    "artifacts/.gitkeep",
    "artifacts/tuning/.gitkeep",
    "data/demo_live/.gitkeep",
    "data/simulated/.gitkeep",
}
GENERATED_PREFIXES = ("artifacts/", "data/demo_live/", "data/simulated/")
SECRET_NAMES = {".env", "secrets.toml"}
LOCAL_ONLY_FILES = {
    "app.py",
    "DigitalTwin_Round2_Plan.md",
    "Digitaltwin project context.md",
    "PIPELINE.md",
    "notes.md",
}


def tracked_files() -> set[str]:
    result = subprocess.run(
        ["git", "ls-files"], cwd=ROOT, check=True, capture_output=True, text=True
    )
    return {line.strip() for line in result.stdout.splitlines() if line.strip()}


def main() -> int:
    problems: list[str] = []
    for relative in REQUIRED_FILES:
        if not (ROOT / relative).is_file():
            problems.append(f"missing required file: {relative}")

    tracked = tracked_files()
    for relative in REQUIRED_FILES:
        if relative not in tracked:
            problems.append(f"required submission file is not tracked: {relative}")
    for relative in sorted(tracked):
        if relative in ALLOWED_GENERATED_PLACEHOLDERS:
            continue
        if relative.startswith(GENERATED_PREFIXES):
            problems.append(f"generated asset is tracked: {relative}")
        if Path(relative).name in SECRET_NAMES:
            problems.append(f"secret file is tracked: {relative}")
        if relative in LOCAL_ONLY_FILES:
            problems.append(f"legacy/internal file is tracked: {relative}")

    for source in sorted(ROOT.rglob("*.py")):
        if any(part in {"venv", ".venv", "env"} for part in source.parts):
            continue
        try:
            py_compile.compile(str(source), doraise=True)
        except py_compile.PyCompileError as exc:
            problems.append(f"Python compile failure: {source.relative_to(ROOT)}: {exc}")

    if problems:
        print("Submission verification failed:")
        for problem in problems:
            print(f"  - {problem}")
        return 1

    print("Submission verification passed.")
    print("  Primary dashboard: app2.py")
    print(f"  Tracked files checked: {len(tracked)}")
    print("  Generated data, weights, tuning output, and secrets are not tracked.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
