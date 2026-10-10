# Experiment 04 — Gaussian Naive Bayes

## Objective
Train a Gaussian Naive Bayes classifier and export the learned class statistics for direct embedded inference.

## Why this experiment
Gaussian Naive Bayes is attractive for TinyML because a fitted model is represented by a small set of means, variances and class priors. Inference can be implemented with a compact mathematical routine instead of a full ML runtime.

## Workflow
1. Generate a reproducible three-feature dataset.
2. Train `GaussianNB` in Python.
3. Evaluate test accuracy.
4. Export class means, variances and priors to a C/C++ header.
5. Compute class log-likelihoods on the microcontroller.

## Run
```bash
python 04-gaussian-naive-bayes/train.py
```

The script creates `embedded/generated_gnb.h`.

## TinyML observation
Model size grows approximately with the number of classes multiplied by the number of features, which makes Gaussian Naive Bayes suitable for many small sensor-classification problems.
