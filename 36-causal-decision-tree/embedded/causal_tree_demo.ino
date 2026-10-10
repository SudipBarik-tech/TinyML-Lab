#include "causal_tree_model.h"

void setup() {
  Serial.begin(115200);
  delay(500);

  float x[2] = {0.75f, 0.20f};
  float effect = estimateTreatmentEffect(x);

  Serial.print("Estimated treatment effect: ");
  Serial.println(effect, 4);
}

void loop() {}
