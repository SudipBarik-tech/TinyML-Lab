# Experiment 23 — Post-Training Pruning

## Objective
Reduce the number of effective neural-network weights and study the effect of sparsity on model size and accuracy.

## Method
A small dense classifier is trained first. TensorFlow Model Optimization is then used to apply magnitude-based pruning during a short fine-tuning stage. The pruning wrappers are removed before TensorFlow Lite conversion.

## Run
```bash
python 23-post-training-pruning/train.py
```

The script reports baseline accuracy, pruned accuracy, final sparsity and exported TensorFlow Lite model size.

## Embedded deployment
The stripped model is converted to TensorFlow Lite and exported to `embedded/pruned_model_data.h`. On-device speed-up depends on whether the target runtime can exploit sparse weights, so sparsity should not automatically be interpreted as lower latency.

## Learning points
- magnitude pruning;
- sparsity schedules;
- accuracy versus compression;
- difference between sparse storage and actual embedded speed-up.
