# 38 — Adaptive Neuro-Fuzzy Inference System (ANFIS)

## Objective

Build a compact first-order Sugeno neuro-fuzzy model that combines fuzzy membership functions with data-fitted consequent equations.

## Method

This experiment uses two inputs with two Gaussian membership functions per input, producing four fuzzy rules. Rule firing strengths are normalised and the first-order Sugeno consequents are fitted with least squares.

## Files

- `train.py` — prepares synthetic regression data, fits the Sugeno consequent parameters and exports all fuzzy parameters.
- `embedded/anfis_demo.ino` — performs fuzzy inference directly in C++.

## Run

```bash
python 38-anfis/train.py
```

Copy the generated `anfis_model.h` beside the sketch before compiling.

## TinyML relevance

Only a few membership parameters and linear consequent coefficients are stored. The embedded implementation requires basic arithmetic and `expf`, making the memory cost easy to inspect.

## Limitation

The membership centres and widths are deliberately fixed to keep the training pipeline transparent. This is a compact educational ANFIS implementation rather than a reproduction of an external ANFIS package.
