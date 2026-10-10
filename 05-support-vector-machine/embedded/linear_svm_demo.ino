#include "generated_svm.h"

void setup() {
  Serial.begin(115200);
  delay(500);

  float sample[4] = {0.25f, -0.15f, 0.65f, 1.05f};
  float score = svmDecision(sample);
  int prediction = predictLinearSVM(sample);

  Serial.print("SVM decision score: ");
  Serial.println(score, 6);
  Serial.print("Linear SVM prediction: ");
  Serial.println(prediction);
}

void loop() {
}
