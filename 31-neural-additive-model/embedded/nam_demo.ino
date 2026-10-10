#include <Chirale_TensorFlowLite.h>
#include "nam_model.h"

#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

constexpr int ARENA_SIZE = 32 * 1024;
alignas(16) uint8_t tensor_arena[ARENA_SIZE];

tflite::AllOpsResolver resolver;
const tflite::Model* model_ptr = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

void setup() {
  Serial.begin(115200);
  delay(500);

  model_ptr = tflite::GetModel(NAM_MODEL);
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

  const float raw[4] = {0.4f, -0.7f, 1.1f, 0.2f};
  for (int i = 0; i < 4; ++i) {
    const float z = (raw[i] - NAM_MEAN[i]) / NAM_SCALE[i];
    int q = (int)roundf(z / input->params.scale) + input->params.zero_point;
    q = constrain(q, -128, 127);
    input->data.int8[i] = (int8_t)q;
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
