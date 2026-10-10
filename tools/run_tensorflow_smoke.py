from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

TF_MARKERS = (
    "import tensorflow",
    "from tensorflow",
    "tensorflow_model_optimization",
)


def uses_tensorflow(path: Path) -> bool:
    source = path.read_text(encoding="utf-8", errors="ignore").lower()
    return any(marker in source for marker in TF_MARKERS)


def main() -> int:
    scripts = [
        p for p in sorted(ROOT.glob("[0-9][0-9]-*/train.py"))
        if uses_tensorflow(p)
    ]
    failures: list[str] = []

    for script in scripts:
        rel = script.relative_to(ROOT)
        print(f"[RUN ] {rel}")
        try:
            result = subprocess.run(
                [sys.executable, str(script)],
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=600,
                check=False,
            )
        except subprocess.TimeoutExpired:
            failures.append(f"{rel}: timed out after 600 seconds")
            print(f"[FAIL] {rel}: timeout")
            continue

        output = result.stdout.strip()
        if output:
            for line in output.splitlines()[-12:]:
                print(f"       {line}")

        if result.returncode != 0:
            failures.append(f"{rel}: exit code {result.returncode}")
            print(f"[FAIL] {rel}: exit code {result.returncode}")
        else:
            print(f"[ OK ] {rel}")

    print("\nSummary")
    print(f"TensorFlow workflows executed: {len(scripts)}")
    if failures:
        print(f"Failures: {len(failures)}")
        for failure in failures:
            print(f" - {failure}")
        return 1

    print("All TensorFlow training/export scripts completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
