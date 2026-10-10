#include "huber_model.h"

void setup() {
  Serial.begin(115200);
  const float sample[HUBER_FEATURES] = {0.25f, -0.40f, 0.80f, 0.10f};
  const float prediction = huberPredict(sample);
  Serial.print("Predicted value: ");
  Serial.println(prediction, 4);
}

void loop() {}
