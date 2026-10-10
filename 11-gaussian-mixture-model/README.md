# Experiment 11 — Gaussian Mixture Model

## Objective
Study probabilistic clustering with a compact Gaussian Mixture Model (GMM) and export the fitted parameters for lightweight inference.

## Method
A two-component GMM is trained on a reproducible two-dimensional synthetic dataset. The Python script reports the fitted component weights, means and diagonal variances, then writes them to `embedded/gmm_model.h`.

The embedded implementation evaluates the log-probability of each Gaussian component and selects the component with the largest score.

## Run
```bash
python 11-gaussian-mixture-model/train.py
```

Then open `embedded/gmm_demo.ino` in the Arduino IDE. Run the training script first so that `gmm_model.h` is generated.

## TinyML relevance
GMM inference is attractive for small anomaly-detection and clustering tasks because the deployed model consists mainly of a few means, variances and mixture weights. The computational cost grows with the number of components and features, so compact configurations are preferred.

## Output to record
- fitted component weights
- component means and variances
- desktop cluster assignments
- device-side cluster prediction
- approximate flash/RAM footprint and inference time

## Limitation
The example uses diagonal covariance matrices to keep embedded computation simple. Full covariance matrices require additional storage and matrix operations.
