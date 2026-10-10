#include "qat_model_data.h"

// Load QAT_MODEL with TensorFlow Lite for Microcontrollers.
// The model expects 8 standardized input features and produces one INT8 output.

void setup() {
  Serial.begin(115200);
  Serial.print("QAT model bytes: ");
  Serial.println(QAT_MODEL_LEN);
}

void loop() {}
