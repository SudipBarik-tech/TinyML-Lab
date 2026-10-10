#include "kmeans_model.h"

void setup() {
  Serial.begin(115200);
  delay(500);

  float sample[KMEANS_FEATURE_COUNT] = {2.1f, -0.7f};
  int cluster = kmeans_predict(sample);

  Serial.print("Assigned cluster: ");
  Serial.println(cluster);
}

void loop() {
}
