#include "knn_model.h"

int predictKNN(const float raw[KNN_F]) {
  float x[KNN_F];
  for (int j = 0; j < KNN_F; ++j) x[j] = (raw[j] - KNN_MEAN[j]) / KNN_SCALE[j];

  float bestDist[KNN_K] = {1e30f, 1e30f, 1e30f};
  int bestLabel[KNN_K] = {-1, -1, -1};

  for (int i = 0; i < KNN_N; ++i) {
    float d = 0.0f;
    for (int j = 0; j < KNN_F; ++j) {
      float e = x[j] - KNN_X[i][j];
      d += e * e;
    }
    for (int k = 0; k < KNN_K; ++k) {
      if (d < bestDist[k]) {
        for (int s = KNN_K - 1; s > k; --s) {
          bestDist[s] = bestDist[s - 1];
          bestLabel[s] = bestLabel[s - 1];
        }
        bestDist[k] = d;
        bestLabel[k] = KNN_Y[i];
        break;
      }
    }
  }

  int votes[3] = {0, 0, 0};
  for (int k = 0; k < KNN_K; ++k) if (bestLabel[k] >= 0 && bestLabel[k] < 3) votes[bestLabel[k]]++;
  int winner = 0;
  for (int c = 1; c < 3; ++c) if (votes[c] > votes[winner]) winner = c;
  return winner;
}

void setup() {
  Serial.begin(115200);
  float sample[KNN_F] = {0.2f, -0.5f, 1.1f, 0.3f};
  Serial.print("Predicted class: ");
  Serial.println(predictKNN(sample));
}

void loop() {}
