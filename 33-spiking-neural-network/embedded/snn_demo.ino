#include <math.h>
#include "snn_model.h"

float sigmoidf(float x) {
  return 1.0f / (1.0f + expf(-x));
}

float classifySpikeWindow(float rate0, float rate1) {
  float membrane[SNN_HIDDEN] = {0};
  float counts[SNN_HIDDEN] = {0};

  for (int t = 0; t < SNN_TIMESTEPS; ++t) {
    const float inputSpike[SNN_INPUTS] = {
      random(10000) < (long)(rate0 * 10000.0f) ? 1.0f : 0.0f,
      random(10000) < (long)(rate1 * 10000.0f) ? 1.0f : 0.0f
    };

    for (int h = 0; h < SNN_HIDDEN; ++h) {
      for (int i = 0; i < SNN_INPUTS; ++i) {
        membrane[h] += inputSpike[i] * SNN_W_IN[i][h];
      }
      if (membrane[h] >= SNN_THRESHOLD[h]) {
        counts[h] += 1.0f;
        membrane[h] -= SNN_THRESHOLD[h];
      }
    }
  }

  float logit = SNN_READOUT_B;
  for (int h = 0; h < SNN_HIDDEN; ++h) {
    logit += counts[h] * SNN_READOUT_W[h];
  }
  return sigmoidf(logit);
}

void setup() {
  Serial.begin(115200);
  randomSeed(42);
}

void loop() {
  // Higher activity on channel 0 should resemble class 0.
  const float probabilityClass1 = classifySpikeWindow(0.22f, 0.07f);
  Serial.print("Class 1 probability: ");
  Serial.println(probabilityClass1, 4);
  delay(1500);
}
