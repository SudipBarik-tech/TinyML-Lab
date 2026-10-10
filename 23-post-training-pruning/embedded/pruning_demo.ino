#include "pruned_model_data.h"

void setup() {
  Serial.begin(115200);
  Serial.print("Pruned model bytes: ");
  Serial.println(PRUNED_MODEL_LEN);
}

void loop() {}
