# Experiment 21 — Huber Regression

## Objective
Study robust linear regression for embedded prediction when a dataset contains occasional outliers.

## Method
The experiment generates a small regression dataset, injects a limited number of large target outliers and compares ordinary linear regression with `HuberRegressor`. The fitted Huber model is exported as a compact coefficient-and-intercept representation.

## Run
```bash
python 21-huber-regression/train.py
```

The script reports MAE and RMSE, then writes `embedded/huber_model.h`.

## Embedded deployment
The device performs one dot product followed by addition of the intercept. This keeps inference memory and latency very small.

## Learning points
- robustness to outliers;
- effect of feature scaling;
- exporting a fitted linear model;
- comparing desktop and microcontroller predictions.
