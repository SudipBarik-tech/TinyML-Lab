#include <TensorFlowLite.h>
#include "bnn_model.h"

#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

constexpr int ARENA_SIZE = 20 * 1024;
uint8_t tensor_arena[ARENA_SIZE];

tflite::AllOpsResolver resolver;
const tflite::Model* model_ptr;
tflite::MicroInterpreter* interpreter;
TfLiteTensor* input;
TfLiteTensor* output;

void setup() {
  Serial.begin(115200);
  model_ptr = tflite::GetModel(BNN_MODEL);

  static tflite::MicroInterpreter static_interpreter(
      model_ptr, resolver, tensor_arena, ARENA_SIZE);
  interpreter = &static_interpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("Tensor allocation failed");
    while (true) {}
  }

  input = interpreter->input(0);
  output = interpreter->output(0);
}

void loop() {
  const float raw[5] = {0.3f, -0.8f, 0.5f, 1.0f, -0.2f};

  for (int i = 0; i < 5; ++i) {
    const float z = (raw[i] - BNN_MEAN[i]) / BNN_SCALE[i];
    int q = (int)roundf(z / BNN_INPUT_SCALE) + BNN_INPUT_ZERO;
    q = constrain(q, -128, 127);
    input->data.int8[i] = (int8_t)q;
  }

  if (interpreter->Invoke() != kTfLiteOk) {
    Serial.println("Inference failed");
    delay(1000);
    return;
  }

  const float probability =
      (output->data.int8[0] - BNN_OUTPUT_ZERO) * BNN_OUTPUT_SCALE;
  Serial.print("Deterministic class 1 probability: ");
  Serial.println(probability, 4);
  delay(1500);
}
