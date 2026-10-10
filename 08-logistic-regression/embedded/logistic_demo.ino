#include "logistic_model.h"

void setup() {
  Serial.begin(115200);
  delay(500);

  float sample[LOGREG_FEATURE_COUNT] = {0.8f, -0.1f, 1.2f, 0.4f};
  float probability = logistic_probability(sample);
  int predictedClass = logistic_predict(sample);

  Serial.print("Probability: ");
  Serial.println(probability, 6);
  Serial.print("Predicted class: ");
  Serial.println(predictedClass);
}

void loop() {
}
