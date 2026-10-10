#include "generated_xgb.h"

void setup() {
  Serial.begin(115200);
  delay(500);

  float sample[4] = {0.10f, -0.25f, 0.80f, 1.15f};
  int prediction = predictXGBoost(sample);

  Serial.print("XGBoost prediction: ");
  Serial.println(prediction);
}

void loop() {
}
