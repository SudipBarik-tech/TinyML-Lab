from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SKIP_MARKERS = (
    "import tensorflow",
    "from tensorflow",
    "tensorflow_model_optimization",
)


def should_skip(path: Path) -> bool:
    source = path.read_text(encoding="utf-8", errors="ignore").lower()
    return any(marker in source for marker in SKIP_MARKERS)


def main() -> int:
    scripts = sorted(ROOT.glob("[0-9][0-9]-*/train.py"))
    failures: list[str] = []
    executed = 0
    skipped = 0

    for script in scripts:
        rel = script.relative_to(ROOT)
        if should_skip(script):
            skipped += 1
            print(f"[SKIP] {rel} (TensorFlow/TFLite workflow)")
            continue

        print(f"[RUN ] {rel}")
        try:
            result = subprocess.run(
                [sys.executable, str(script)],
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=180,
                check=False,
            )
        except subprocess.TimeoutExpired:
            failures.append(f"{rel}: timed out after 180 seconds")
            print(f"[FAIL] {rel}: timeout")
            continue

        executed += 1
        output = result.stdout.strip()
        if output:
            lines = output.splitlines()
            for line in lines[-8:]:
                print(f"       {line}")

        if result.returncode != 0:
            failures.append(f"{rel}: exit code {result.returncode}")
            print(f"[FAIL] {rel}: exit code {result.returncode}")
        else:
            print(f"[ OK ] {rel}")

    print("\nSummary")
    print(f"Executed: {executed}")
    print(f"Skipped TensorFlow workflows: {skipped}")
    if failures:
        print(f"Failures: {len(failures)}")
        for failure in failures:
            print(f" - {failure}")
        return 1

    print("All lightweight training scripts completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
