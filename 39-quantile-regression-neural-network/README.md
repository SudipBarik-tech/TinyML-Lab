# 39 — Quantile Regression Neural Network

## Objective

Predict multiple conditional quantiles instead of only a single mean value, providing a lightweight estimate of uncertainty or prediction intervals.

## Method

A compact neural network receives two input features and predicts the 0.10, 0.50 and 0.90 quantiles. Training uses the pinball loss for each target quantile.

## Files

- `train.py` — creates heteroscedastic regression data, trains the network and exports dense-layer weights.
- `embedded/qrnn_demo.ino` — performs direct C++ inference without TensorFlow Lite.

## Run

```bash
python 39-quantile-regression-neural-network/train.py
```

Copy the generated `qrnn_model.h` beside the sketch before compiling.

## TinyML relevance

The deployed network contains one small ReLU hidden layer and three outputs. Direct weight export keeps the inference path simple and allows RAM and flash use to be inspected without a runtime interpreter.

## Observation

The lower and upper quantiles form an interval around the median estimate. Quantile crossing should be checked during evaluation because independently trained quantile outputs are not guaranteed to remain ordered.
