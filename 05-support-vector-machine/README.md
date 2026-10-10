# Experiment 05 — Linear Support Vector Machine

## Objective
Train a compact linear Support Vector Machine and export its weight vector and bias for direct microcontroller inference.

## Why this experiment
A linear SVM is highly suitable for embedded deployment because prediction reduces to a dot product and a threshold comparison. It avoids storing support vectors and kernel calculations required by nonlinear SVMs.

## Workflow
1. Generate a reproducible four-feature binary-classification dataset.
2. Standardise features with `StandardScaler`.
3. Train `LinearSVC`.
4. Evaluate test accuracy.
5. Export scaler parameters, weights and bias to C/C++.
6. Run the generated model from the Arduino sketch.

## Run
```bash
python 05-support-vector-machine/train.py
```

The script creates `embedded/generated_svm.h`.

## TinyML observation
Linear SVM inference is extremely small, but its usefulness depends on whether the classes can be separated well in the selected feature space. Feature engineering is therefore especially important.
