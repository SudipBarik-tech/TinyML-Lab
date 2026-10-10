#include "student_model_data.h"

void setup() {
  Serial.begin(115200);
  Serial.print("Distilled student model bytes: ");
  Serial.println(STUDENT_MODEL_LEN);
}

void loop() {}
