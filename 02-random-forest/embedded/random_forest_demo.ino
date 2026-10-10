#include "generated_forest.h"

void setup() {
  Serial.begin(115200);
  delay(500);

  float sample[3] = {0.20f, -0.35f, 1.10f};
  int prediction = predictRandomForest(sample);

  Serial.print("Random Forest prediction: ");
  Serial.println(prediction);
}

void loop() {
}
