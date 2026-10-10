# 36 — Causal Decision Tree

## Objective

Estimate how a treatment effect changes across input conditions using a small tree that can be exported to an embedded target.

## Method

This experiment generates a synthetic randomized-treatment dataset with two covariates. A transformed outcome is used as a simple estimate of conditional treatment effect, and a shallow `DecisionTreeRegressor` learns interpretable subgroups.

The tree predicts an estimated treatment effect rather than an ordinary class label.

## Files

- `train.py` — generates data, fits the treatment-effect tree and exports C++ rules.
- `embedded/causal_tree_demo.ino` — evaluates the exported treatment-effect estimate.

## Run

```bash
python 36-causal-decision-tree/train.py
```

Copy the generated `causal_tree_model.h` beside the Arduino sketch before compiling.

## Embedded relevance

A shallow tree becomes nested comparisons, so inference requires little RAM and has predictable latency.

## Important limitation

This is a compact educational causal-tree experiment on randomized synthetic data. It is not a substitute for a full causal-inference workflow. Real observational studies require careful treatment of confounding, overlap, uncertainty and study design.
