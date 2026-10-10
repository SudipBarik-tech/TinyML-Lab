# Validation Status

This document records what has actually been checked for the current TinyML-Lab implementation. It separates desktop-side validation from hardware validation so that unmeasured embedded results are not presented as verified results.

## Current validation summary

| Check | Status |
|---|---|
| 40 experiment folders present | Passed |
| Required README, `train.py` and embedded sketch structure | Passed |
| Python syntax compilation for all 40 training scripts | Passed |
| Training/export contract between `train.py` and local Arduino model header | Passed |
| Non-TensorFlow training/export workflows | 28/28 passed |
| TensorFlow/TFLite training/export workflows | 12/12 passed |
| Legacy repository path / stale source-branding scan | Passed |
| Arduino/ESP32 compilation | Pending hardware/toolchain validation |
| Desktop-to-device prediction equivalence | Pending hardware validation |
| Flash, RAM and inference-latency measurements | Pending hardware validation |

## Automated checks

Three GitHub Actions workflows are used during validation.

### Repository checks

`repository-checks.yml` performs static validation on every push to `main` and on pull requests. It checks the 40-folder structure, compiles every Python source file, verifies the model-header naming contract between the training scripts and Arduino examples, and checks for stale legacy repository markers.

### Lightweight runtime checks

`lightweight-runtime.yml` installs NumPy, scikit-learn, XGBoost, pandas and matplotlib, then executes every non-TensorFlow experiment. The current suite executes 28 training/export scripts and requires each process to finish successfully.

### TensorFlow runtime checks

`tensorflow-runtime.yml` installs TensorFlow and TensorFlow Model Optimization and executes the remaining 12 neural-network workflows. This includes training and the model-export step used by each experiment.

The successful CI environment uses Python 3.11. The runtime checks currently exercise recent releases of NumPy, scikit-learn, XGBoost and TensorFlow rather than relying only on syntax inspection.

## Issues found during validation

The CI pass was useful because it exposed problems that a visual code review would not reliably catch:

1. The K-Means C-header exporter contained an invalid Python f-string expression. The exporter was rewritten to build the centre rows before formatting the header.
2. XGBoost changed the serialized representation of `base_score` to a bracketed value in the tested release. The regression exporter now parses the numeric value defensively.
3. The original fixed-length LSTM conversion produced dynamic TensorList operations that could not be lowered to a built-in-only TFLite model. The LSTM is now explicitly unrolled for the fixed sequence length, allowing the INT8 conversion workflow to complete.

After these corrections, the static suite, the 28 lightweight runtime workflows and all 12 TensorFlow/TFLite workflows complete successfully in GitHub Actions.

## What "passed" means

A successful desktop runtime check means that the experiment can create its dataset, train or fit the intended model, calculate its reported desktop metric, and complete its export step without a Python runtime error in the CI environment.

For experiments that produce C/C++ headers, the generated data or functions are created by the training script. For TensorFlow experiments, the check also covers TensorFlow Lite conversion where that conversion is part of the experiment.

This does **not** mean that every generated model has already been flashed to a microcontroller.

## Hardware validation still required

Before publishing measured TinyML deployment results, each selected target should be built with the actual Arduino/ESP32 toolchain and tested on the intended board. The hardware pass should record at least:

- target board and core/toolchain version;
- sketch flash usage;
- static and peak RAM where measurable;
- model or parameter storage size;
- inference latency over repeated runs;
- desktop prediction versus device prediction for fixed test vectors;
- any operator or library limitation observed on the board.

No RAM, flash or latency figure should be treated as measured until this hardware stage is completed.

## Re-running validation locally

Static repository checks:

```bash
python tools/validate_repository.py
```

Non-TensorFlow runtime checks:

```bash
python tools/run_lightweight_smoke.py
```

TensorFlow runtime checks require the dependencies from `requirements.txt` and can be run with:

```bash
python tools/run_tensorflow_smoke.py
```

The GitHub Actions workflows remain the reference automated environment for the repository-wide checks.
