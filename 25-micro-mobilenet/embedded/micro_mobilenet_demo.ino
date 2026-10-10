#include <Chirale_TensorFlowLite.h>
#include "micro_mobilenet_model.h"

#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

constexpr int TENSOR_ARENA_SIZE = 96 * 1024;
alignas(16) uint8_t tensorArena[TENSOR_ARENA_SIZE];

tflite::AllOpsResolver resolver;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

void setup() {
  Serial.begin(115200);
  delay(500);

  model = tflite::GetModel(MICRO_MOBILENET_MODEL);
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

  // Build a simple vertical-bar image resembling class 0 from training.
  for (int row = 0; row < 16; ++row) {
    for (int col = 0; col < 16; ++col) {
      float value = 0.0f;
      if (row >= 3 && row < 13 && col >= 7 && col < 9) {
        value = 1.0f;
      }
      const int index = row * 16 + col;
      int q = (int)roundf(value / input->params.scale) + input->params.zero_point;
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
  for (int c = 1; c < 3; ++c) {
    if (output->data.int8[c] > bestScore) {
      bestScore = output->data.int8[c];
      bestClass = c;
    }
  }

  Serial.print("Predicted image class: ");
  Serial.println(bestClass);
}

void loop() {}
