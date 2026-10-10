#include "sefr_model.h"

void setup() {
  Serial.begin(115200);
  delay(500);

  float sample[SEFR_FEATURE_COUNT] = {1.2f, -0.4f, 0.8f, 1.0f};
  int predictedClass = sefr_predict(sample);

  Serial.print("Predicted class: ");
  Serial.println(predictedClass);
}

void loop() {
}
