#include "pca_model.h"

void setup() {
  Serial.begin(115200);
  delay(500);

  float sample[PCA_INPUT_DIM] = {1.0f, 0.5f, -0.2f, 1.4f};
  float projected[PCA_OUTPUT_DIM];
  pca_transform(sample, projected);

  Serial.print("PC1: ");
  Serial.println(projected[0], 6);
  Serial.print("PC2: ");
  Serial.println(projected[1], 6);
}

void loop() {
}
