#include "xgb_regressor_model.h"

void setup() {
  Serial.begin(115200);
  delay(500);

  float sample[4] = {0.20f, -0.35f, 0.80f, 0.15f};
  float prediction = xgbRegress(sample);

  Serial.print("Predicted value: ");
  Serial.println(prediction, 5);
}

void loop() {
}
