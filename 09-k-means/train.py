from pathlib import Path

from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

RANDOM_STATE = 42


def export_header(model: KMeans, output: Path):
    centers = model.cluster_centers_
    rows = []
    for row in centers:
        rows.append("{" + ", ".join(f"{v:.8f}f" for v in row) + "}")

    text = f"""#pragma once

const int KMEANS_CLUSTER_COUNT = {centers.shape[0]};
const int KMEANS_FEATURE_COUNT = {centers.shape[1]};
const float KMEANS_CENTERS[KMEANS_CLUSTER_COUNT][KMEANS_FEATURE_COUNT] = {{
  {',\n  '.join(rows)}
}};

inline int kmeans_predict(const float *x) {{
  int bestCluster = 0;
  float bestDistance = 3.4028235e+38F;

  for (int c = 0; c < KMEANS_CLUSTER_COUNT; ++c) {{
    float distance = 0.0f;
    for (int i = 0; i < KMEANS_FEATURE_COUNT; ++i) {{
      float d = x[i] - KMEANS_CENTERS[c][i];
      distance += d * d;
    }}
    if (distance < bestDistance) {{
      bestDistance = distance;
      bestCluster = c;
    }}
  }}
  return bestCluster;
}}
"""
    output.write_text(text, encoding="utf-8")


def main():
    X, _ = make_blobs(
        n_samples=600,
        centers=3,
        n_features=2,
        cluster_std=0.9,
        random_state=RANDOM_STATE,
    )

    model = KMeans(n_clusters=3, n_init=10, random_state=RANDOM_STATE)
    model.fit(X)

    print("Cluster centers:")
    for i, center in enumerate(model.cluster_centers_):
        print(f"  {i}: {center}")

    out = Path(__file__).parent / "embedded" / "kmeans_model.h"
    out.parent.mkdir(exist_ok=True)
    export_header(model, out)
    print(f"Exported model: {out}")


if __name__ == "__main__":
    main()
