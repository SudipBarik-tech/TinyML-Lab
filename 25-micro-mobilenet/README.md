# Experiment 25 — Micro MobileNet

## Objective
Build a very small depthwise-separable convolutional network suitable for studying MobileNet-style efficiency on constrained devices.

## Method
The model uses depthwise convolution followed by pointwise convolution, global average pooling and a compact classifier. A small synthetic image dataset is used to keep the experiment self-contained and reproducible.

## Run
```bash
python 25-micro-mobilenet/train.py
```

The training script reports validation accuracy, converts the model to full INT8 TensorFlow Lite and exports `embedded/micro_mobilenet_model.h`.

## Embedded deployment
The generated model is intended for TensorFlow Lite for Microcontrollers. The experiment emphasises parameter count, model size and the cost of depthwise-separable convolution compared with a conventional CNN.

## Learning points
- depthwise and pointwise convolution;
- compact CNN design;
- INT8 conversion;
- model-size and deployment trade-offs.
