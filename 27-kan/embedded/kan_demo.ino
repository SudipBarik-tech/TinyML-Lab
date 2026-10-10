#include "kan_model.h"

void setup() {
  Serial.begin(115200);
  delay(500);

  float sample[KAN_FEATURES] = {0.5f, 0.2f};
  Serial.print("KAN-inspired score: ");
  Serial.println(kanScore(sample), 6);
  Serial.print("Predicted class: ");
  Serial.println(kanPredict(sample));
}

void loop() {
  delay(1000);
}
