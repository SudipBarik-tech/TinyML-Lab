# Experiment 26 — Multinomial Naive Bayes

## Objective

Implement a compact Multinomial Naive Bayes classifier for count-based features and export the fitted log-probabilities for microcontroller inference.

## Why this experiment matters

Multinomial Naive Bayes is useful when the input is represented as non-negative counts, such as word counts, event counts or histogram features. Its inference stage is very small: one prior term plus a weighted sum of feature counts for each class.

## Workflow

1. Generate a reproducible three-class count dataset.
2. Train `MultinomialNB` in scikit-learn.
3. Evaluate test accuracy.
4. Export class log-priors and feature log-probabilities to `embedded/multinomial_nb_model.h`.
5. Reproduce the same class-score calculation in Arduino/C++.

## Files

- `train.py` — dataset generation, training, evaluation and header export.
- `embedded/multinomial_nb_demo.ino` — device-side inference example.
- `embedded/multinomial_nb_model.h` — generated after running the training script.

## Run

```bash
python 26-multinomial-naive-bayes/train.py
```

## Embedded inference

For class `c`, the score is

```text
score(c) = log P(c) + Σ x_j log P(feature_j | c)
```

The predicted class is the one with the maximum score. No dynamic memory is required.

## TinyML observation

The method is attractive for small count-based classifiers because model size grows only with the number of classes and input features. The main restriction is that input values should be non-negative and meaningfully represent counts or frequencies.
