#include <TensorFlowLite.h>
#include "lstm_model.h"

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

constexpr int TENSOR_ARENA_SIZE = 48 * 1024;
uint8_t tensorArena[TENSOR_ARENA_SIZE];

tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

void setup() {
  Serial.begin(115200);

  const tflite::Model* model = tflite::GetModel(LSTM_MODEL);
  static tflite::MicroMutableOpResolver<6> resolver;
  resolver.AddUnidirectionalSequenceLSTM();
  resolver.AddFullyConnected();
  resolver.AddSoftmax();
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
  Serial.println("LSTM model ready");
}

void loop() {
  // Fill input->data.int8 with one quantised [20 x 3] sequence,
  // then call interpreter->Invoke() and read output->data.int8.
  delay(1000);
}
