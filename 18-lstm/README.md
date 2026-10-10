# 18 — LSTM Sequence Classification

This experiment uses a small LSTM network to classify short time-series windows and convert the trained model to an INT8 TensorFlow Lite representation for embedded deployment.

## Objective
Demonstrate a reproducible sequence-model workflow suitable for sensor-oriented TinyML experiments.

## Method
A synthetic three-class sequence dataset is generated with a fixed seed. A compact Keras LSTM model is trained, evaluated and converted with post-training integer quantisation using a representative dataset.

## Run
```bash
python 18-lstm/train.py
```

The script writes `embedded/lstm_model.h` containing the quantised `.tflite` model bytes.

## Embedded inference
`embedded/lstm_demo.ino` shows the TensorFlow Lite Micro setup pattern. The generated header can be used on ESP32/ESP32-S3 or another supported microcontroller.

## TinyML note
LSTM models can be useful for IMU and temporal sensor data, but recurrent operations are more expensive than simple classical models. Model size, tensor arena size and inference latency should therefore be measured on the actual target board.
