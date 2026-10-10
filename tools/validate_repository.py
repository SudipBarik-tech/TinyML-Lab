from __future__ import annotations

import py_compile
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

EXPECTED = {
    1: "01-decision-tree",
    2: "02-random-forest",
    3: "03-xgboost",
    4: "04-gaussian-naive-bayes",
    5: "05-support-vector-machine",
    6: "06-sefr",
    7: "07-pca",
    8: "08-logistic-regression",
    9: "09-k-means",
    10: "10-linear-regression",
    11: "11-gaussian-mixture-model",
    12: "12-mlp",
    13: "13-cnn",
    14: "14-xgboost-regression",
    15: "15-poisson-regression",
    16: "16-knn",
    17: "17-elastic-net",
    18: "18-lstm",
    19: "19-autoencoder",
    20: "20-q-learning",
    21: "21-huber-regression",
    22: "22-quantization-aware-training",
    23: "23-post-training-pruning",
    24: "24-knowledge-distillation",
    25: "25-micro-mobilenet",
    26: "26-multinomial-naive-bayes",
    27: "27-kan",
    28: "28-lda",
    29: "29-rbf-network",
    30: "30-gcn",
    31: "31-neural-additive-model",
    32: "32-bayesian-neural-network",
    33: "33-spiking-neural-network",
    34: "34-recurrent-neural-network",
    35: "35-liquid-neural-network",
    36: "36-causal-decision-tree",
    37: "37-generative-adversarial-network",
    38: "38-anfis",
    39: "39-quantile-regression-neural-network",
    40: "40-self-organising-map",
}

# Split the legacy strings so the validator does not flag its own source file.
OLD_LOGIN = "thommas" + "kevin"
OLD_NAME = "thommas" + " kevin"
OLD_ROOT = "tinyml-" + "main/"
FORBIDDEN_MARKERS = (OLD_LOGIN, OLD_NAME, OLD_ROOT)

TEXT_SUFFIXES = {".py", ".ino", ".h", ".hpp", ".md", ".txt", ".yml", ".yaml"}


def fail(errors: list[str], message: str) -> None:
    errors.append(message)
    print(f"[FAIL] {message}")


def ok(message: str) -> None:
    print(f"[ OK ] {message}")


def validate_structure(errors: list[str]) -> None:
    for number, folder_name in EXPECTED.items():
        folder = ROOT / folder_name
        if not folder.is_dir():
            fail(errors, f"Experiment {number:02d}: missing folder {folder_name}")
            continue

        required = [folder / "README.md", folder / "train.py", folder / "embedded"]
        for item in required:
            if not item.exists():
                fail(errors, f"{folder_name}: missing {item.relative_to(ROOT)}")

        embedded = folder / "embedded"
        if embedded.is_dir() and not any(embedded.glob("*.ino")):
            fail(errors, f"{folder_name}: no Arduino .ino example in embedded/")

    present = sorted(
        p.name for p in ROOT.iterdir()
        if p.is_dir() and re.match(r"^\d{2}-", p.name)
    )
    expected_names = sorted(EXPECTED.values())
    unexpected = sorted(set(present) - set(expected_names))
    missing = sorted(set(expected_names) - set(present))
    if unexpected:
        fail(errors, f"Unexpected numbered experiment folders: {unexpected}")
    if missing:
        fail(errors, f"Missing numbered experiment folders: {missing}")
    if not unexpected and not missing:
        ok("All 40 numbered experiment folders are present")


def validate_python(errors: list[str]) -> None:
    files = sorted(ROOT.glob("[0-9][0-9]-*/train.py"))
    for path in files:
        try:
            py_compile.compile(str(path), doraise=True)
        except py_compile.PyCompileError as exc:
            fail(errors, f"Python syntax error in {path.relative_to(ROOT)}: {exc.msg}")
    if len(files) == 40 and not any("syntax error" in e.lower() for e in errors):
        ok("All 40 training scripts pass Python syntax compilation")


def validate_text(errors: list[str]) -> None:
    findings: list[str] = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        if ".git" in path.parts:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore").lower()
        except OSError:
            continue
        for marker in FORBIDDEN_MARKERS:
            if marker in text:
                findings.append(f"{path.relative_to(ROOT)} contains a legacy source marker")
    for finding in findings:
        fail(errors, finding)
    if not findings:
        ok("No stale upstream branding or legacy repository path found")


def validate_root_docs(errors: list[str]) -> None:
    for name in ("README.md", "requirements.txt", ".gitignore", "docs/EXPERIMENT_TEMPLATE.md"):
        if not (ROOT / name).exists():
            fail(errors, f"Missing root project file: {name}")
    readme = ROOT / "README.md"
    if readme.exists():
        text = readme.read_text(encoding="utf-8", errors="ignore")
        for folder_name in EXPECTED.values():
            if folder_name not in text:
                fail(errors, f"Main README does not list {folder_name}")


def main() -> int:
    errors: list[str] = []
    print("TinyML-Lab static repository validation\n")
    validate_structure(errors)
    validate_python(errors)
    validate_text(errors)
    validate_root_docs(errors)

    print("\nSummary")
    if errors:
        print(f"Validation failed with {len(errors)} issue(s).")
        return 1
    print("Validation passed.")
    print("Note: this is a static check; it does not train models or compile Arduino sketches.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
