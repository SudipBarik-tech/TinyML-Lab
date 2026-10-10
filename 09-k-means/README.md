# 09 — K-Means Clustering

## Objective

Demonstrate an unsupervised learning workflow that can be reproduced on an embedded device using only the final cluster centroids.

## Method

A two-dimensional synthetic dataset is generated and fitted with K-Means. The learned centroids are exported to a compact C header. Embedded inference assigns a new sample to the nearest centroid using squared Euclidean distance.

## Run

```bash
python 09-k-means/train.py
```

The script prints the learned cluster centres and writes `embedded/kmeans_model.h`.

## Embedded inference

Only the cluster centroids are required at runtime. For each input sample the device computes the distance to every centroid and returns the index of the nearest one.

## Files

- `train.py` — dataset generation, clustering and model export
- `embedded/kmeans_demo.ino` — Arduino-compatible clustering example

## Observation

K-Means is attractive for embedded use because inference is simple and does not require a large runtime library. Cost grows with the number of clusters and input dimensions.
