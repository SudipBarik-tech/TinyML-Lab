#include "rbf_model.h"

void setup() {
  Serial.begin(115200);
  delay(500);

  float sample[RBF_FEATURES] = {0.6f, 0.1f};
  Serial.print("RBF score: ");
  Serial.println(rbfScore(sample), 6);
  Serial.print("Predicted class: ");
  Serial.println(rbfPredict(sample));
}

void loop() {
  delay(1000);
}
