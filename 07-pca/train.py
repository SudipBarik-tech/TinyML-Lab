from pathlib import Path

import numpy as np
from sklearn.datasets import make_classification
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42


def export_header(scaler: StandardScaler, pca: PCA, output: Path):
    scale = ", ".join(f"{v:.8f}f" for v in scaler.scale_)
    scaler_mean = ", ".join(f"{v:.8f}f" for v in scaler.mean_)
    pca_mean = ", ".join(f"{v:.8f}f" for v in pca.mean_)

    component_rows = []
    for row in pca.components_:
        component_rows.append("{" + ", ".join(f"{v:.8f}f" for v in row) + "}")
    components = ",\n  ".join(component_rows)

    text = f"""#pragma once

const int PCA_INPUT_DIM = {len(scaler.mean_)};
const int PCA_OUTPUT_DIM = {pca.n_components_};

const float PCA_SCALER_MEAN[PCA_INPUT_DIM] = {{{scaler_mean}}};
const float PCA_SCALER_SCALE[PCA_INPUT_DIM] = {{{scale}}};
const float PCA_MEAN[PCA_INPUT_DIM] = {{{pca_mean}}};
const float PCA_COMPONENTS[PCA_OUTPUT_DIM][PCA_INPUT_DIM] = {{
  {components}
}};

inline void pca_transform(const float *x, float *z) {{
  float standardized[PCA_INPUT_DIM];
  for (int i = 0; i < PCA_INPUT_DIM; ++i) {{
    standardized[i] = (x[i] - PCA_SCALER_MEAN[i]) / PCA_SCALER_SCALE[i];
  }}

  for (int c = 0; c < PCA_OUTPUT_DIM; ++c) {{
    float value = 0.0f;
    for (int i = 0; i < PCA_INPUT_DIM; ++i) {{
      value += (standardized[i] - PCA_MEAN[i]) * PCA_COMPONENTS[c][i];
    }}
    z[c] = value;
  }}
}}
"""
    output.write_text(text, encoding="utf-8")


def main():
    X, _ = make_classification(
        n_samples=700,
        n_features=4,
        n_informative=4,
        n_redundant=0,
        random_state=RANDOM_STATE,
    )

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    pca.fit(X_scaled)

    print("Explained variance ratio:", np.round(pca.explained_variance_ratio_, 4))
    print(f"Total retained variance: {pca.explained_variance_ratio_.sum():.4f}")

    out = Path(__file__).parent / "embedded" / "pca_model.h"
    out.parent.mkdir(exist_ok=True)
    export_header(scaler, pca, out)
    print(f"Exported transformation: {out}")


if __name__ == "__main__":
    main()
