#include "linear_regression_model.h"

void setup() {
  Serial.begin(115200);
  delay(500);

  float sample[LINREG_FEATURE_COUNT] = {0.5f, -1.0f, 1.2f, 0.3f};
  float prediction = linear_regression_predict(sample);

  Serial.print("Predicted value: ");
  Serial.println(prediction, 6);
}

void loop() {
}
