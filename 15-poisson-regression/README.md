# Experiment 15 — Poisson Regression

## Objective
Study a compact count-prediction model that can be exported as a small set of coefficients for embedded inference.

## Method
A reproducible synthetic count dataset is generated from a Poisson process. A `PoissonRegressor` is trained in Python using standardised features. The fitted coefficients, intercept and scaling parameters are exported to `embedded/poisson_model.h`.

The embedded prediction computes the linear response and applies the exponential link function.

## Run
```bash
python 15-poisson-regression/train.py
```

Then open `embedded/poisson_demo.ino` after the generated header is available.

## TinyML relevance
Poisson regression is useful when the target is a non-negative event count, such as packet arrivals, machine faults or repeated detections. Deployment is compact because only the learned coefficients and feature-scaling values are required.

## Output to record
- mean absolute error
- mean Poisson deviance
- learned coefficients
- desktop and device prediction comparison
- inference latency

## Limitation
Poisson regression assumes the conditional target mean is positive and uses a log link. It is not appropriate for arbitrary continuous targets.
