# Experiment 01 — Decision Tree

## Objective
Train a compact decision-tree classifier in Python and export its decision rules as a small C/C++ header for microcontroller inference.

## Why this experiment
Decision trees are useful for TinyML because inference requires only comparisons and branches. A shallow tree can run without a floating-point ML runtime and is easy to inspect.

## Workflow
1. Generate a reproducible two-feature classification dataset.
2. Split the data into training and test sets.
3. Train a depth-limited `DecisionTreeClassifier`.
4. Evaluate accuracy on unseen samples.
5. Export the fitted tree to `embedded/generated_model.h`.
6. Compile `embedded/decision_tree_demo.ino` in Arduino IDE or PlatformIO.

## Run
```bash
python 01-decision-tree/train.py
```

The script creates the model header used by the embedded sketch.

## Files
```text
01-decision-tree/
├── README.md
├── train.py
└── embedded/
    └── decision_tree_demo.ino
```

## TinyML observation
A shallow tree provides deterministic latency and very small inference code. The main trade-off is that reducing tree depth can reduce classification accuracy.

## Reproducibility
The random seed, train/test split and model depth are fixed in `train.py`, so the experiment can be regenerated consistently.
