#include "gan_generator_model.h"

void setup() {
  Serial.begin(115200);
  delay(500);

  float z[GAN_LATENT] = {0.25f, -0.70f, 1.10f};
  float sample[2];
  ganGenerate(z, sample);

  Serial.print("Generated sample: ");
  Serial.print(sample[0], 4);
  Serial.print(", ");
  Serial.println(sample[1], 4);
}

void loop() {}
