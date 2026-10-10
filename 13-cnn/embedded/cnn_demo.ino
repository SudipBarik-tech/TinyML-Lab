#include <Chirale_TensorFlowLite.h>
#include "cnn_model_data.h"

#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

constexpr int kTensorArenaSize = 32 * 1024;
alignas(16) uint8_t tensorArena[kTensorArenaSize];

tflite::AllOpsResolver resolver;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* inputTensor = nullptr;
TfLiteTensor* outputTensor = nullptr;

void setup() {
  Serial.begin(115200);
  delay(500);

  model = tflite::GetModel(CNN_MODEL_DATA);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("TFLite schema mismatch");
    return;
  }

  static tflite::MicroInterpreter staticInterpreter(
      model, resolver, tensorArena, kTensorArenaSize);
  interpreter = &staticInterpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("Tensor allocation failed");
    return;
  }

  inputTensor = interpreter->input(0);
  outputTensor = interpreter->output(0);

  // Example class-0 signal: 3 cycles over a 64-sample window.
  for (int i = 0; i < 64; ++i) {
    const float x = sinf(2.0f * PI * 3.0f * i / 64.0f);
    int q = (int)roundf(x / inputTensor->params.scale) + inputTensor->params.zero_point;
    q = constrain(q, -128, 127);
    inputTensor->data.int8[i] = (int8_t)q;
  }

  if (interpreter->Invoke() != kTfLiteOk) {
    Serial.println("Inference failed");
    return;
  }

  int bestClass = 0;
  int8_t bestScore = outputTensor->data.int8[0];
  for (int i = 1; i < 2; ++i) {
    if (outputTensor->data.int8[i] > bestScore) {
      bestScore = outputTensor->data.int8[i];
      bestClass = i;
    }
  }

  Serial.print("Predicted class: ");
  Serial.println(bestClass);
}

void loop() {}
