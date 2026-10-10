#include "qrnn_model.h"

void setup() {
  Serial.begin(115200);
  delay(500);

  float x[QRNN_INPUTS] = {0.80f, -0.40f};
  float q[QRNN_OUTPUTS];
  qrnnPredict(x, q);

  Serial.print("q10: ");
  Serial.println(q[0], 4);
  Serial.print("q50: ");
  Serial.println(q[1], 4);
  Serial.print("q90: ");
  Serial.println(q[2], 4);
}

void loop() {}
