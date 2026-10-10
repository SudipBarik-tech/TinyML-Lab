#include "micro_mobilenet_model.h"

void setup() {
  Serial.begin(115200);
  Serial.print("Micro MobileNet model bytes: ");
  Serial.println(MICRO_MOBILENET_MODEL_LEN);
}

void loop() {}
