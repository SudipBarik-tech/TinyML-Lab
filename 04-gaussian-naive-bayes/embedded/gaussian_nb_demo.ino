#include "generated_gnb.h"

void setup() {
  Serial.begin(115200);
  delay(500);

  float sample[3] = {0.40f, -0.10f, 0.95f};
  int prediction = predictGaussianNB(sample);

  Serial.print("GaussianNB prediction: ");
  Serial.println(prediction);
}

void loop() {
}
