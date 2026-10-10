#include <Chirale_TensorFlowLite.h>
#include "qat_model_data.h"

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

  model = tflite::GetModel(QAT_MODEL);
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

  // The training pipeline standardises the eight features before inference.
  const float standardized[8] = {0.25f, -0.40f, 0.15f, 0.80f, -0.35f, 0.10f, 0.55f, -0.20f};
  for (int i = 0; i < 8; ++i) {
    int q = (int)roundf(standardized[i] / input->params.scale) + input->params.zero_point;
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
