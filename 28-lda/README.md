# Experiment 28 — Linear Discriminant Analysis

## Objective

Train a multiclass Linear Discriminant Analysis classifier and export its compact linear decision rule for embedded inference.

## Method

1. Use the Wine dataset from scikit-learn.
2. Split the data with a fixed random seed.
3. Train LDA using the `lsqr` solver.
4. Evaluate multiclass accuracy.
5. Export class coefficients and intercepts.
6. Reproduce the class-score calculation on Arduino/ESP32.

## Files

- `train.py` — training, evaluation and model export.
- `embedded/lda_model.h` — generated coefficient header.
- `embedded/lda_demo.ino` — device-side inference example.

## Run

```bash
python 28-lda/train.py
```

## Embedded inference

For each class, the device calculates a linear discriminant score:

```text
score_c = bias_c + w_c · x
```

The class with the largest score is selected.

## TinyML observation

LDA is attractive for embedded classification because inference can be reduced to multiply-accumulate operations with no tree traversal or neural runtime. Model size grows approximately with the number of classes multiplied by the number of input features.
