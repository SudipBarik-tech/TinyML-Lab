#include "generated_model.h"

void setup() {
  Serial.begin(115200);
  delay(500);

  float sample[2] = {0.35f, -0.80f};
  int prediction = predictDecisionTree(sample);

  Serial.print("Decision Tree prediction: ");
  Serial.println(prediction);
}

void loop() {
}
