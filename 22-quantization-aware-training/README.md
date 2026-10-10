# Experiment 22 — Quantization-Aware Training

## Objective
Train a small neural network while simulating low-precision arithmetic so that the final INT8 model retains good accuracy after conversion.

## Method
A compact dense classifier is first trained in floating point. TensorFlow Model Optimization then inserts fake-quantization operations during fine-tuning. The final model is converted to a fully integer TensorFlow Lite model.

## Run
```bash
python 22-quantization-aware-training/train.py
```

The script reports floating-point and quantized accuracy and exports `embedded/qat_model_data.h`.

## Embedded deployment
The generated model is intended for TensorFlow Lite for Microcontrollers on ESP32-S3 or another supported microcontroller.

## Learning points
- why post-training quantization may reduce accuracy;
- how QAT adapts weights to quantization noise;
- INT8 conversion and representative datasets;
- model-size and accuracy trade-offs.
