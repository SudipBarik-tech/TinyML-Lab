# Experiment 30 — Tiny Graph Convolutional Network

## Objective

Implement a very small Graph Convolutional Network (GCN) from first principles and study what is required to execute graph inference on a constrained device.

## Scope

This experiment uses one small fixed graph for node classification. The aim is to make the graph-convolution operation transparent and exportable, not to reproduce a large graph-learning framework.

## Method

1. Define a small undirected graph, node features and node labels.
2. Add self-loops and compute the symmetric normalised adjacency matrix.
3. Train a two-layer GCN in NumPy using gradient descent.
4. Evaluate node-classification accuracy.
5. Export the normalised adjacency matrix and learned weights to C/C++.
6. Run the same two graph-convolution stages in an Arduino/ESP32 sketch.

The forward path is

```text
H = ReLU(A_norm X W0)
Z = A_norm H W1
```

and the class with the highest logit is selected for each node.

## Files

- `train.py` — NumPy GCN training, evaluation and header export.
- `embedded/gcn_model.h` — generated graph and model parameters.
- `embedded/gcn_demo.ino` — fixed-graph embedded inference example.

## Run

```bash
python 30-gcn/train.py
```

## TinyML observation

GCNs are less naturally suited to small microcontrollers than ordinary vector classifiers because memory and computation depend on the graph as well as model weights. A fixed, small graph can still be practical. Dynamic or large graphs require more careful sparse storage, streaming or edge/cloud partitioning.
