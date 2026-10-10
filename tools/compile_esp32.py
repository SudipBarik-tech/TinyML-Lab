from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FQBN = "esp32:esp32:esp32"

TF_MARKERS = (
    "import tensorflow",
    "from tensorflow",
    "tensorflow_model_optimization",
)


def uses_tensorflow(path: Path) -> bool:
    source = path.read_text(encoding="utf-8", errors="ignore").lower()
    return any(marker in source for marker in TF_MARKERS)


def compile_sketch(sketch: Path) -> tuple[int, str]:
    with tempfile.TemporaryDirectory(prefix="tinyml_esp32_") as tmp:
        tmp_root = Path(tmp)
        sketch_name = sketch.stem
        sketch_dir = tmp_root / sketch_name
        sketch_dir.mkdir()

        shutil.copy2(sketch, sketch_dir / f"{sketch_name}.ino")
        for header in sketch.parent.glob("*.h"):
            shutil.copy2(header, sketch_dir / header.name)
        for header in sketch.parent.glob("*.hpp"):
            shutil.copy2(header, sketch_dir / header.name)

        result = subprocess.run(
            [
                "arduino-cli",
                "compile",
                "--fqbn",
                FQBN,
                "--warnings",
                "all",
                str(sketch_dir),
            ],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
            timeout=300,
        )
        return result.returncode, result.stdout


def main() -> int:
    failures: list[str] = []
    compiled = 0
    skipped = 0

    for train in sorted(ROOT.glob("[0-9][0-9]-*/train.py")):
        folder = train.parent
        rel_folder = folder.relative_to(ROOT)

        if uses_tensorflow(train):
            skipped += 1
            print(f"[SKIP] {rel_folder} (requires TensorFlow Lite Micro integration)")
            continue

        for sketch in sorted((folder / "embedded").glob("*.ino")):
            rel = sketch.relative_to(ROOT)
            print(f"[BUILD] {rel}")
            try:
                returncode, output = compile_sketch(sketch)
            except subprocess.TimeoutExpired:
                failures.append(f"{rel}: compile timeout")
                print(f"[FAIL] {rel}: compile timeout")
                continue

            compiled += 1
            if returncode != 0:
                failures.append(str(rel))
                print(f"[FAIL] {rel}")
                for line in output.splitlines()[-30:]:
                    print(f"       {line}")
            else:
                print(f"[ OK ] {rel}")

    print("\nSummary")
    print(f"ESP32 sketches compiled: {compiled}")
    print(f"TFLM experiments skipped: {skipped}")

    if failures:
        print(f"Failures: {len(failures)}")
        for failure in failures:
            print(f" - {failure}")
        return 1

    print("All non-TFLM embedded examples compiled with the generic ESP32 Arduino target.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
