#include "poisson_model.h"

void setup() {
  Serial.begin(115200);
  delay(500);

  float sample[3] = {0.40f, -0.20f, 0.70f};
  float expectedCount = poissonPredict(sample);

  Serial.print("Predicted expected count: ");
  Serial.println(expectedCount, 5);
}

void loop() {
}
