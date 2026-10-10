# Experiment 02 — Random Forest

## Objective
Train a small Random Forest classifier and export each tree as plain C/C++ decision logic for embedded inference.

## Why this experiment
Random Forest improves robustness compared with a single tree, but the memory cost grows with the number and depth of trees. This experiment keeps both values intentionally small so that the trade-off can be observed on embedded hardware.

## Workflow
1. Generate a reproducible three-feature classification dataset.
2. Train a compact `RandomForestClassifier`.
3. Measure test accuracy.
4. Export each tree as a C/C++ function.
5. Combine tree outputs using majority voting.
6. Run the exported model from the Arduino sketch.

## Run
```bash
python 02-random-forest/train.py
```

The script writes `embedded/generated_forest.h`.

## TinyML observation
Increasing the number of trees can improve stability, but also increases flash usage and inference operations. For microcontrollers, a small forest is often more practical than a desktop-sized model.
