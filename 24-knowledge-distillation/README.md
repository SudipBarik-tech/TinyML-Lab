# Experiment 24 — Knowledge Distillation

## Objective
Train a compact student network using guidance from a larger teacher model, then prepare the student for embedded inference.

## Method
The teacher is trained on a synthetic multi-class dataset. A smaller student network is then trained using a weighted combination of normal classification loss and softened teacher predictions.

## Run
```bash
python 24-knowledge-distillation/train.py
```

The script reports teacher and student accuracy, converts the student to INT8 TensorFlow Lite and exports `embedded/student_model_data.h`.

## Embedded deployment
Only the student model is deployed. The teacher is used during training and does not consume microcontroller memory.

## Learning points
- teacher/student architecture design;
- temperature-scaled soft targets;
- balancing hard-label and distillation losses;
- reducing deployed model size while retaining useful accuracy.
