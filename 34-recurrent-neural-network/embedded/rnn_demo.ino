#include <Chirale_TensorFlowLite.h>
#include "rnn_model.h"

#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

constexpr int ARENA_SIZE = 48 * 1024;
alignas(16) uint8_t tensor_arena[ARENA_SIZE];

tflite::AllOpsResolver resolver;
const tflite::Model* model_ptr = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

void setup() {
  Serial.begin(115200);
  delay(500);

  model_ptr = tflite::GetModel(RNN_MODEL);
  if (model_ptr->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("TFLite schema mismatch");
    return;
  }

  static tflite::MicroInterpreter static_interpreter(
      model_ptr, resolver, tensor_arena, ARENA_SIZE);
  interpreter = &static_interpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("Tensor allocation failed");
    return;
  }

  input = interpreter->input(0);
  output = interpreter->output(0);

  for (int t = 0; t < RNN_TIMESTEPS; ++t) {
    const float angle = (2.0f * PI * t) / (RNN_TIMESTEPS - 1);
    const float sample[RNN_FEATURES] = {
      sinf(angle),
      sinf(angle + 0.25f)
    };

    for (int f = 0; f < RNN_FEATURES; ++f) {
      const int index = t * RNN_FEATURES + f;
      int q = (int)roundf(sample[f] / input->params.scale) + input->params.zero_point;
      q = constrain(q, -128, 127);
      input->data.int8[index] = (int8_t)q;
    }
  }

  if (interpreter->Invoke() != kTfLiteOk) {
    Serial.println("Inference failed");
    return;
  }

  const float probability =
      (output->data.int8[0] - output->params.zero_point) * output->params.scale;
  Serial.print("Class 1 probability: ");
  Serial.println(probability, 4);
}

void loop() {}
