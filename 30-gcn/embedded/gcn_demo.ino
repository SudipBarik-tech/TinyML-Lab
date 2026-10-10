#include "gcn_model.h"

void setup() {
  Serial.begin(115200);
  delay(500);

  int labels[GCN_NODES];
  gcnPredict(labels);

  for (int i = 0; i < GCN_NODES; ++i) {
    Serial.print("Node ");
    Serial.print(i);
    Serial.print(" -> class ");
    Serial.println(labels[i]);
  }
}

void loop() {
  delay(1000);
}
