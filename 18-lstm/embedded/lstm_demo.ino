#include <Chirale_TensorFlowLite.h>
#include "lstm_model.h"

#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

constexpr int TENSOR_ARENA_SIZE = 64 * 1024;
alignas(16) uint8_t tensorArena[TENSOR_ARENA_SIZE];

tflite::AllOpsResolver resolver;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

void setup() {
  Serial.begin(115200);
  delay(500);

  model = tflite::GetModel(LSTM_MODEL);
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

  // Generate a deterministic 3-channel sequence similar to class 0.
  for (int t = 0; t < LSTM_STEPS; ++t) {
    const float angle = (2.0f * PI * t) / LSTM_STEPS;
    const float sample[LSTM_FEATURES] = {
      sinf(angle),
      cosf(angle),
      0.25f
    };

    for (int f = 0; f < LSTM_FEATURES; ++f) {
      const int index = t * LSTM_FEATURES + f;
      int q = (int)roundf(sample[f] / input->params.scale) + input->params.zero_point;
      q = constrain(q, -128, 127);
      input->data.int8[index] = (int8_t)q;
    }
  }

  if (interpreter->Invoke() != kTfLiteOk) {
    Serial.println("Inference failed");
    return;
  }

  int bestClass = 0;
  int8_t bestScore = output->data.int8[0];
  for (int c = 1; c < LSTM_CLASSES; ++c) {
    if (output->data.int8[c] > bestScore) {
      bestScore = output->data.int8[c];
      bestClass = c;
    }
  }

  Serial.print("Predicted sequence class: ");
  Serial.println(bestClass);
}

void loop() {}
