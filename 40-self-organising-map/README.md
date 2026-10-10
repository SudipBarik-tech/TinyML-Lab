# 40 — Self-Organising Map

## Objective

Train a small Self-Organising Map (SOM) that compresses multidimensional samples into a two-dimensional prototype grid and can be queried on an embedded device.

## Method

A 5 × 5 SOM is trained on synthetic three-dimensional clustered data. For every training sample, the best matching unit is found and nearby prototypes are updated using a Gaussian neighbourhood function.

## Files

- `train.py` — trains the SOM and exports the codebook vectors.
- `embedded/som_demo.ino` — finds the best matching unit using Euclidean distance.

## Run

```bash
python 40-self-organising-map/train.py
```

Copy the generated `som_model.h` beside the sketch before compiling.

## TinyML relevance

Deployment requires only the trained prototype vectors and a nearest-prototype search. No training logic is required on the microcontroller.

## Observation

The output grid coordinate provides a compact topological representation of the input. For larger maps, flash use and search cost increase linearly with the number of stored prototypes.
