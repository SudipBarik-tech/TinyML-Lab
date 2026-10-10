#include "anfis_model.h"

void setup() {
  Serial.begin(115200);
  delay(500);

  float x[2] = {0.60f, -0.35f};
  float y = anfisPredict(x);

  Serial.print("ANFIS output: ");
  Serial.println(y, 5);
}

void loop() {}
