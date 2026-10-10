#include <Chirale_TensorFlowLite.h>
#include "autoencoder_model.h"

#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

constexpr int TENSOR_ARENA_SIZE = 32 * 1024;
alignas(16) uint8_t tensorArena[TENSOR_ARENA_SIZE];

tflite::AllOpsResolver resolver;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

void setup() {
  Serial.begin(115200);
  delay(500);

  model = tflite::GetModel(AUTOENCODER_MODEL);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("TFLite schema mismatch");
    return;
  }

  static tflite::MicroInterpreter staticInterpreter(
      model, resolver, tensorArena, TENSOR_ARENA_SIZE);
  interpreter = &staticInterpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("Tensor allocation failed");
    return;
  }

  input = interpreter->input(0);
  output = interpreter->output(0);

  const float raw[AE_FEATURES] = {0.30f, 0.10f, -0.20f, 0.15f, 0.25f, -0.10f};
  float standardized[AE_FEATURES];

  for (int i = 0; i < AE_FEATURES; ++i) {
    standardized[i] = (raw[i] - AE_MEAN[i]) / AE_STD[i];
    int q = (int)roundf(standardized[i] / input->params.scale) + input->params.zero_point;
    q = constrain(q, -128, 127);
    input->data.int8[i] = (int8_t)q;
  }

  if (interpreter->Invoke() != kTfLiteOk) {
    Serial.println("Inference failed");
    return;
  }

  float mse = 0.0f;
  for (int i = 0; i < AE_FEATURES; ++i) {
    const float reconstructed =
        (output->data.int8[i] - output->params.zero_point) * output->params.scale;
    const float error = standardized[i] - reconstructed;
    mse += error * error;
  }
  mse /= AE_FEATURES;

  Serial.print("Reconstruction MSE: ");
  Serial.println(mse, 6);
  Serial.print("Anomaly: ");
  Serial.println(mse > AE_THRESHOLD ? "yes" : "no");
}

void loop() {}
