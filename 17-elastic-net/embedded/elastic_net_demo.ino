#include "elastic_net_model.h"

float predictElasticNet(const float raw[EN_FEATURES]) {
  float y = EN_INTERCEPT;
  for (int i = 0; i < EN_FEATURES; ++i) {
    float x = (raw[i] - EN_MEAN[i]) / EN_SCALE[i];
    y += EN_COEF[i] * x;
  }
  return y;
}

void setup() {
  Serial.begin(115200);
  float sample[EN_FEATURES] = {0.3f, -0.2f, 0.8f, 1.1f, -0.4f, 0.5f};
  Serial.print("Prediction: ");
  Serial.println(predictElasticNet(sample), 6);
}

void loop() {}
