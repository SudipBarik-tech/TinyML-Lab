#include "som_model.h"

void setup() {
  Serial.begin(115200);
  delay(500);

  float x[SOM_FEATURES] = {0.40f, -1.10f, 1.30f};
  int bmu = somBestMatchingUnit(x);
  int row = bmu / SOM_COLS;
  int col = bmu % SOM_COLS;

  Serial.print("BMU row: ");
  Serial.println(row);
  Serial.print("BMU col: ");
  Serial.println(col);
}

void loop() {}
