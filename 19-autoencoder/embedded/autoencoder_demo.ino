#include <TensorFlowLite.h>
#include "autoencoder_model.h"

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

constexpr int TENSOR_ARENA_SIZE = 24 * 1024;
uint8_t tensorArena[TENSOR_ARENA_SIZE];

tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

void setup() {
  Serial.begin(115200);
  const tflite::Model* model = tflite::GetModel(AUTOENCODER_MODEL);

  static tflite::MicroMutableOpResolver<5> resolver;
  resolver.AddFullyConnected();
  resolver.AddRelu();
  resolver.AddReshape();
  resolver.AddQuantize();
  resolver.AddDequantize();

  static tflite::MicroInterpreter staticInterpreter(
      model, resolver, tensorArena, TENSOR_ARENA_SIZE);
  interpreter = &staticInterpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("Tensor allocation failed");
    return;
  }

  input = interpreter->input(0);
  output = interpreter->output(0);
  Serial.println("Autoencoder ready");
}

void loop() {
  // 1. Standardise the raw features with AE_MEAN and AE_STD.
  // 2. Quantise them using input->params.scale and zero_point.
  // 3. Invoke the model.
  // 4. Dequantise the reconstruction and calculate mean squared error.
  // 5. Flag the sample if the reconstruction error exceeds AE_THRESHOLD.
  delay(1000);
}
