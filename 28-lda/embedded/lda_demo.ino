#include "lda_model.h"

void setup() {
  Serial.begin(115200);
  delay(500);

  // Example Wine feature vector (13 features).
  float sample[LDA_FEATURES] = {
    13.2f, 1.78f, 2.14f, 11.2f, 100.0f, 2.65f, 2.76f,
    0.26f, 1.28f, 4.38f, 1.05f, 3.40f, 1050.0f
  };

  Serial.print("Predicted class: ");
  Serial.println(ldaPredict(sample));
}

void loop() {
  delay(1000);
}
