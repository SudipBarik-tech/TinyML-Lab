#include <math.h>
#include "liquid_model.h"

float sigmoidf(float x) {
  return 1.0f / (1.0f + expf(-x));
}

float classifySequence(const float *sequence) {
  float h[LIQUID_HIDDEN] = {0};
  float previous[LIQUID_HIDDEN] = {0};

  for (int t = 0; t < LIQUID_TIMESTEPS; ++t) {
    for (int i = 0; i < LIQUID_HIDDEN; ++i) {
      previous[i] = h[i];
    }

    for (int i = 0; i < LIQUID_HIDDEN; ++i) {
      float pre = LIQUID_W_IN[i] * sequence[t] + LIQUID_BIAS[i];
      for (int j = 0; j < LIQUID_HIDDEN; ++j) {
        pre += LIQUID_W_REC[i][j] * previous[j];
      }
      const float candidate = tanhf(pre);
      h[i] = (1.0f - LIQUID_ALPHA[i]) * previous[i]
           + LIQUID_ALPHA[i] * candidate;
    }
  }

  float logit = LIQUID_READOUT_B;
  for (int i = 0; i < LIQUID_HIDDEN; ++i) {
    logit += LIQUID_READOUT_W[i] * h[i];
  }
  return sigmoidf(logit);
}

void setup() {
  Serial.begin(115200);
}

void loop() {
  float sequence[LIQUID_TIMESTEPS];
  for (int t = 0; t < LIQUID_TIMESTEPS; ++t) {
    const float u = (float)t / (LIQUID_TIMESTEPS - 1);
    sequence[t] = 0.45f * sinf(2.0f * PI * u)
                + (0.70f * u - 0.35f);
  }

  const float probabilityClass1 = classifySequence(sequence);
  Serial.print("Class 1 probability: ");
  Serial.println(probabilityClass1, 4);
  delay(1500);
}
