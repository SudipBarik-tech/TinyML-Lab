from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

TF_MARKERS = (
    "import tensorflow",
    "from tensorflow",
    "tensorflow_model_optimization",
)

ARDUINO_STUB = r'''
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

#ifndef PI
#define PI 3.1415926535897932384626433832795
#endif

class SerialStub {
public:
  void begin(unsigned long) {}

  template <typename T>
  void print(const T&) {}

  template <typename T, typename U>
  void print(const T&, const U&) {}

  template <typename T>
  void println(const T&) {}

  template <typename T, typename U>
  void println(const T&, const U&) {}

  void println() {}
};

static SerialStub Serial;

inline void delay(unsigned long) {}
inline unsigned long millis() { return 0UL; }
inline unsigned long micros() { return 0UL; }
inline void randomSeed(unsigned long) {}
inline long random(long upper) { return upper > 0 ? 0L : 0L; }
inline long random(long lower, long upper) {
  (void)upper;
  return lower;
}
'''


def uses_tensorflow(path: Path) -> bool:
    source = path.read_text(encoding="utf-8", errors="ignore").lower()
    return any(marker in source for marker in TF_MARKERS)


def main() -> int:
    failures: list[str] = []
    compiled = 0
    skipped = 0

    for train in sorted(ROOT.glob("[0-9][0-9]-*/train.py")):
        folder = train.parent
        rel_folder = folder.relative_to(ROOT)

        if uses_tensorflow(train):
            skipped += 1
            print(f"[SKIP] {rel_folder} (external TensorFlow Lite Micro runtime)")
            continue

        sketches = sorted((folder / "embedded").glob("*.ino"))
        for sketch in sketches:
            wrapper_text = ARDUINO_STUB + f'\n#include "{sketch.name}"\n'
            with tempfile.TemporaryDirectory(prefix="tinyml_cpp_") as tmp:
                wrapper = Path(tmp) / "sketch_check.cpp"
                wrapper.write_text(wrapper_text, encoding="utf-8")

                cmd = [
                    "g++",
                    "-std=c++17",
                    "-Wall",
                    "-Wextra",
                    "-pedantic",
                    "-fsyntax-only",
                    "-I",
                    str(sketch.parent),
                    str(wrapper),
                ]
                result = subprocess.run(
                    cmd,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    check=False,
                )

            compiled += 1
            rel = sketch.relative_to(ROOT)
            if result.returncode != 0:
                failures.append(str(rel))
                print(f"[FAIL] {rel}")
                if result.stdout.strip():
                    for line in result.stdout.splitlines()[-20:]:
                        print(f"       {line}")
            else:
                print(f"[ OK ] {rel}")

    print("\nSummary")
    print(f"Embedded sketches syntax-checked: {compiled}")
    print(f"TensorFlow/TFLM experiments skipped: {skipped}")

    if failures:
        print(f"Failures: {len(failures)}")
        for item in failures:
            print(f" - {item}")
        return 1

    print("All non-TensorFlow embedded examples passed host-side C++ syntax checks.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
