#include "multinomial_nb_model.h"

void setup() {
  Serial.begin(115200);
  delay(500);

  float sample[MNB_FEATURES] = {7, 1, 1, 0, 2, 1};
  int predicted = multinomialNBPredict(sample);

  Serial.print("Predicted class: ");
  Serial.println(predicted);
}

void loop() {
  delay(1000);
}
