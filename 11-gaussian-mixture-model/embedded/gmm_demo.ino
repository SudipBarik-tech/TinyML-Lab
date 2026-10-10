#include "gmm_model.h"

void setup() {
  Serial.begin(115200);
  delay(500);

  float sample[2] = {1.2f, 1.0f};
  int cluster = gmmPredict(sample);

  Serial.print("Predicted GMM component: ");
  Serial.println(cluster);
}

void loop() {
}
