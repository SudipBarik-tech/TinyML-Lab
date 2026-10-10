# Experiment 03 — XGBoost Classifier

## Objective
Train a very small XGBoost classifier and export its boosted trees as plain C/C++ functions for lightweight inference.

## Why this experiment
Boosted trees can provide strong accuracy with a small number of shallow estimators. The experiment deliberately restricts tree depth and estimator count to keep the exported model understandable and suitable for embedded study.

## Workflow
1. Generate a reproducible binary-classification dataset.
2. Train a compact `XGBClassifier`.
3. Measure test accuracy.
4. Read the trained booster as JSON trees.
5. Convert the tree rules to C/C++.
6. Perform binary classification from the accumulated tree score.

## Run
```bash
python 03-xgboost/train.py
```

The script generates `embedded/generated_xgb.h`.

## TinyML observation
The important embedded parameters are the number of trees, maximum depth and number of features. Smaller boosted models can be practical even when a full desktop XGBoost model would be too large for a microcontroller.
