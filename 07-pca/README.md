# 07 — Principal Component Analysis

## Objective

Study PCA as a lightweight feature-reduction stage before embedded inference. The experiment reduces a four-dimensional input to two principal components and exports the fitted transformation to C/C++.

## Method

A reproducible synthetic dataset is standardised and fitted with a two-component PCA model. The exported header stores the training mean, feature scale and PCA components so the same transformation can be reproduced on a microcontroller.

## Run

```bash
python 07-pca/train.py
```

The script prints the explained-variance ratio and writes `embedded/pca_model.h`.

## Embedded inference

For each input sample, the device:

1. standardises the original features using training statistics;
2. subtracts the PCA mean in standardised space;
3. computes two dot products to obtain the principal-component coordinates.

## Files

- `train.py` — fitting, evaluation summary and header export
- `embedded/pca_demo.ino` — Arduino-compatible projection example

## Observation

PCA does not perform classification by itself. Its TinyML value is in reducing feature dimensionality so that a later classifier can use fewer inputs and potentially less memory and computation.
