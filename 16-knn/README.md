# 16 — K-Nearest Neighbours

This experiment studies a small KNN classifier for embedded inference.

## Objective
Train a compact KNN model and deploy the reference samples directly to an Arduino/ESP32 target.

## Method
A synthetic three-class, four-feature dataset is generated with a fixed seed. Features are standardised before fitting a `KNeighborsClassifier` with `k=3`. To keep the embedded footprint controlled, a reduced prototype set is selected from the training data and exported to `embedded/knn_model.h`.

## Run
```bash
python 16-knn/train.py
```

The script reports desktop accuracy and writes the prototype vectors, labels, scaling parameters and K value to the generated header.

## Embedded inference
`embedded/knn_demo.ino` standardises an input sample, calculates squared Euclidean distances to the stored prototypes and predicts by majority vote among the three nearest neighbours.

## TinyML note
KNN requires no model fitting at inference time, but memory and latency grow with the number of stored reference points. Prototype reduction is therefore an important deployment decision.
