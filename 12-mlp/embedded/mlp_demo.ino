#include "mlp_model.h"

void setup() {
  Serial.begin(115200);
  delay(500);

  float sample[2] = {0.45f, 0.70f};
  float probability = mlpProbability(sample);
  int predictedClass = mlpPredict(sample);

  Serial.print("Probability: ");
  Serial.println(probability, 5);
  Serial.print("Predicted class: ");
  Serial.println(predictedClass);
}

void loop() {
}
