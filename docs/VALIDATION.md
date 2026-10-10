# Validation Status

This document records what has actually been checked for the current TinyML-Lab implementation. It separates desktop-side validation from physical-device validation so that unmeasured embedded results are not presented as verified results.

## Current validation summary

| Check | Status |
|---|---|
| 40 experiment folders present | Passed |
| Required README, `train.py` and embedded sketch structure | Passed |
| Python syntax compilation for all 40 training scripts | Passed |
| Training/export contract between `train.py` and local Arduino model header | Passed |
| Non-TensorFlow training/export workflows | 28/28 passed |
| TensorFlow/TFLite training/export workflows | 12/12 passed |
| Host-side C++ syntax check for non-TFLM embedded examples | 28/28 passed |
| Generic ESP32 Arduino CLI compilation for non-TFLM examples | 28/28 passed |
| Legacy repository path / stale source-branding scan | Passed |
| TFLM sketch compilation with a selected embedded library package | Pending |
| Desktop-to-device prediction equivalence | Pending physical-device validation |
| Flash, RAM and inference-latency measurements | Pending physical-device validation |

## Automated checks

Four GitHub Actions workflows are used during validation.

### Repository checks

`repository-checks.yml` performs static validation on pushes to `main` and on pull requests. It checks the 40-folder structure, compiles every Python source file, verifies the model-header naming contract between the training scripts and Arduino examples, and checks for stale legacy repository markers.

### Lightweight runtime checks

`lightweight-runtime.yml` installs NumPy, scikit-learn, XGBoost, pandas and matplotlib, then executes every non-TensorFlow experiment. The current suite executes 28 training/export scripts and requires each process to finish successfully.

After those exporters run, `tools/compile_embedded_stubs.py` performs a host-side C++17 syntax check on the 28 corresponding embedded sketches using a small compatibility stub for common Arduino functions. This catches C/C++ syntax and generated-header integration errors without pretending to be a board build.

### TensorFlow runtime checks

`tensorflow-runtime.yml` installs TensorFlow and TensorFlow Model Optimization and executes the remaining 12 neural-network workflows. This includes training and the model-export step used by each experiment.

### ESP32 compile checks

`esp32-compile.yml` installs Arduino CLI and the Espressif Arduino core, regenerates the non-TensorFlow model headers, and compiles all 28 corresponding sketches with the generic ESP32 target `esp32:esp32:esp32`.

The current successful build used **ESP32 Arduino core 3.3.12**. All 28 non-TFLM sketches compiled successfully. The 12 TensorFlow Lite Micro examples are intentionally kept out of this generic compile job because their library packaging and include paths depend on the selected TFLM/Arduino integration.

## Issues found during validation

The CI passes exposed problems that a visual code review would not reliably catch:

1. The K-Means C-header exporter contained an invalid Python f-string expression. The exporter was rewritten to build the centre rows before formatting the header.
2. XGBoost changed the serialized representation of `base_score` to a bracketed value in the tested release. The regression exporter now parses the numeric value defensively.
3. The original fixed-length LSTM conversion produced dynamic TensorList operations that could not be lowered to a built-in-only TFLite model. The LSTM is now explicitly unrolled for the fixed sequence length, allowing the INT8 conversion workflow to complete.
4. The embedded host checker was expanded to model Arduino's `PI`, formatted `Serial.print/println`, `randomSeed` and `random` interfaces before using host compilation as a meaningful syntax gate.

After these corrections, the static suite, all 28 lightweight runtime workflows, all 12 TensorFlow/TFLite workflows, all 28 non-TFLM host-side embedded syntax checks and all 28 generic ESP32 Arduino builds complete successfully.

## What "passed" means

A successful desktop runtime check means that the experiment can create its dataset, train or fit the intended model, calculate its reported desktop metric, and complete its export step without a Python runtime error in the CI environment.

For experiments that produce C/C++ headers, the generated data or functions are created by the training script. For TensorFlow experiments, the check also covers TensorFlow Lite conversion where that conversion is part of the experiment.

A successful generic ESP32 compile means that the non-TFLM sketch and its generated header compile with Arduino CLI for `esp32:esp32:esp32` using Espressif Arduino core 3.3.12. It verifies the build path, but it is not evidence that the program has been flashed, timed or functionally tested on a physical board.

## Physical-device validation still required

Before publishing measured TinyML deployment results, selected targets should be flashed and tested on the intended hardware. The hardware pass should record at least:

- target board and core/toolchain version;
- sketch flash usage;
- static and peak RAM where measurable;
- model or parameter storage size;
- inference latency over repeated runs;
- desktop prediction versus device prediction for fixed test vectors;
- any operator or library limitation observed on the board.

For the 12 TensorFlow Lite Micro experiments, the selected Arduino/TFLM package should also be fixed and documented before claiming board-level compilation.

No RAM, flash or latency figure should be treated as measured until this physical-device stage is completed.

## Re-running validation locally

Static repository checks:

```bash
python tools/validate_repository.py
```

Non-TensorFlow runtime checks:

```bash
python tools/run_lightweight_smoke.py
```

After the non-TensorFlow exporters have generated their headers, run the host-side embedded syntax checks with:

```bash
python tools/compile_embedded_stubs.py
```

TensorFlow runtime checks require the dependencies from `requirements.txt` and can be run with:

```bash
python tools/run_tensorflow_smoke.py
```

With Arduino CLI and the ESP32 core installed, the non-TFLM ESP32 builds can be repeated with:

```bash
python tools/compile_esp32.py
```

The GitHub Actions workflows remain the reference automated environment for the repository-wide checks.
