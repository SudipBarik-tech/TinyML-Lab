# 19 — Autoencoder for Lightweight Anomaly Detection

This experiment uses a small autoencoder to learn the normal structure of low-dimensional sensor-like data and detect unusual samples from reconstruction error.

## Objective
Build a compact anomaly-detection workflow that can be transferred to a constrained device.

## Method
A reproducible synthetic normal dataset is generated from correlated features. A small dense autoencoder is trained only on normal samples. Separate anomalous samples are created for evaluation. The reconstruction-error threshold is derived from the training distribution.

## Run
```bash
python 19-autoencoder/train.py
```

The script prints reconstruction-error statistics, converts the network to an INT8 TensorFlow Lite model and writes `embedded/autoencoder_model.h`.

## Embedded inference
`embedded/autoencoder_demo.ino` provides the TensorFlow Lite Micro setup pattern. A deployment should reconstruct the input, compute mean squared reconstruction error and compare it with the exported anomaly threshold.

## TinyML note
Autoencoders are useful when abnormal examples are rare, but the threshold must be calibrated for the actual sensor and operating environment.
