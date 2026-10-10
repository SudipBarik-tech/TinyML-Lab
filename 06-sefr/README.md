# 06 — SEFR Classifier

## Objective

Implement a compact binary classifier suitable for very small embedded systems. SEFR (Scalable, Efficient and Fast Classifier) uses one weight per input feature and a single decision threshold, making inference inexpensive in memory and computation.

## Method

This experiment uses a synthetic two-class dataset with four numerical features. During training, the mean of every feature is computed separately for the positive and negative classes. A normalised feature weight is derived from the separation between those class means. The projected training samples are then used to select a threshold that maximises classification accuracy.

## Run

```bash
python 06-sefr/train.py
```

The script prints the test accuracy and writes `embedded/sefr_model.h`.

## Embedded inference

The generated model requires only:

1. a weighted sum of the input features;
2. one comparison with the learned threshold.

This makes the method useful when flash, RAM and inference time are tightly constrained.

## Files

- `train.py` — dataset generation, training, evaluation and C header export
- `embedded/sefr_demo.ino` — Arduino-compatible inference example
- `embedded/sefr_model.h` — generated after running the training script

## Observation

SEFR is deliberately simple. Its strength is low deployment cost rather than modelling highly nonlinear decision boundaries. For a practical TinyML comparison, its accuracy should be considered together with model size and inference latency.
