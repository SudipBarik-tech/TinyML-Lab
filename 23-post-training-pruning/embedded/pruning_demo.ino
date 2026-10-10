#include <Chirale_TensorFlowLite.h>
#include "pruned_model_data.h"

#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

constexpr int TENSOR_ARENA_SIZE = 40 * 1024;
alignas(16) uint8_t tensorArena[TENSOR_ARENA_SIZE];

tflite::AllOpsResolver resolver;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

void setup() {
  Serial.begin(115200);
  delay(500);

  model = tflite::GetModel(PRUNED_MODEL);
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

  // Dynamic-range conversion keeps the model input/output as float32.
  const float standardized[10] = {
    0.20f, -0.35f, 0.60f, -0.10f, 0.45f,
    -0.55f, 0.15f, 0.30f, -0.25f, 0.05f
  };
  for (int i = 0; i < 10; ++i) {
    input->data.f[i] = standardized[i];
  }

  if (interpreter->Invoke() != kTfLiteOk) {
    Serial.println("Inference failed");
    return;
  }

  Serial.print("Class 1 probability: ");
  Serial.println(output->data.f[0], 4);
}

void loop() {}
