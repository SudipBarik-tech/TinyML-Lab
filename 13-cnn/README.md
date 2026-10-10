# Experiment 13 — Compact 1D Convolutional Neural Network

## Objective
Build a small convolutional neural network for classifying short one-dimensional sensor-like signals and convert it to an integer TensorFlow Lite model for embedded deployment.

## Method
Synthetic sine-wave segments with two frequency classes are generated locally, avoiding external dataset dependencies. A compact Conv1D network is trained with TensorFlow/Keras, then converted with full integer quantisation. The converted `.tflite` model is also written as a C header for TensorFlow Lite for Microcontrollers.

## Run
```bash
python 13-cnn/train.py
```

Generated files:
- `tiny_cnn_int8.tflite`
- `embedded/cnn_model_data.h`

## TinyML relevance
Convolutional layers can learn local patterns from IMU, vibration, audio and other time-series signals. Integer quantisation reduces model size and makes inference more practical on small embedded devices.

## Embedded note
`embedded/cnn_demo.ino` shows the TensorFlow Lite Micro integration pattern. The exact library include paths and tensor-arena size can vary by Arduino core and TFLM package version.

## Output to record
- validation accuracy
- `.tflite` model size
- input/output quantisation parameters
- tensor-arena usage
- measured inference latency on the target board
