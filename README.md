# TinyML Lab

Practical experiments in **embedded machine learning, TinyML and resource-efficient AI**.

This repository is a hands-on academic workspace for studying how machine-learning models move from desktop training environments to constrained embedded devices. The focus is on reproducible training, compact model export, device-side inference and measurable deployment trade-offs.

## Tracks

- **Embedded ML Fundamentals** — compact classical ML models and feature-processing methods.
- **Tiny Neural Networks** — small neural architectures for constrained inference.
- **TinyML Optimisation** — quantisation, pruning, distillation and compact architectures.
- **Advanced Edge AI** — exploratory methods whose deployment feasibility is evaluated rather than assumed.

## Implemented experiments

| No. | Experiment | Training | Embedded export |
|---:|---|:---:|:---:|
| 01 | Decision Tree | ✅ | ✅ |
| 02 | Random Forest | ✅ | ✅ |
| 03 | XGBoost Classifier | ✅ | ✅ |
| 04 | Gaussian Naive Bayes | ✅ | ✅ |
| 05 | Linear Support Vector Machine | ✅ | ✅ |
| 06 | SEFR Classifier | ✅ | ✅ |
| 07 | Principal Component Analysis | ✅ | ✅ |
| 08 | Logistic Regression | ✅ | ✅ |
| 09 | K-Means Clustering | ✅ | ✅ |
| 10 | Linear Regression | ✅ | ✅ |
| 11 | Gaussian Mixture Model | ✅ | ✅ |
| 12 | Multilayer Perceptron | ✅ | ✅ |
| 13 | Compact 1D CNN | ✅ | ✅ |
| 14 | XGBoost Regression | ✅ | ✅ |
| 15 | Poisson Regression | ✅ | ✅ |
| 16 | K-Nearest Neighbours | ✅ | ✅ |
| 17 | Elastic Net Regression | ✅ | ✅ |
| 18 | LSTM Sequence Classifier | ✅ | ✅ |
| 19 | Autoencoder Anomaly Detection | ✅ | ✅ |
| 20 | Q-Learning | ✅ | ✅ |
| 21 | Huber Regression | ✅ | ✅ |
| 22 | Quantization-Aware Training | ✅ | ✅ |
| 23 | Post-Training Pruning | ✅ | ✅ |
| 24 | Knowledge Distillation | ✅ | ✅ |
| 25 | Micro MobileNet | ✅ | ✅ |
| 26 | Multinomial Naive Bayes | ✅ | ✅ |
| 27 | Compact KAN-Inspired Classifier | ✅ | ✅ |
| 28 | Linear Discriminant Analysis | ✅ | ✅ |
| 29 | Radial Basis Function Network | ✅ | ✅ |
| 30 | Tiny Graph Convolutional Network | ✅ | ✅ |
| 31 | Neural Additive Model | ✅ | ✅ |
| 32 | Bayesian NN via MC Dropout | ✅ | ✅ |
| 33 | Spiking Neural Network | ✅ | ✅ |
| 34 | Recurrent Neural Network | ✅ | ✅ |
| 35 | Liquid-Inspired Recurrent Network | ✅ | ✅ |
| 36 | Causal Decision Tree | ✅ | ✅ |
| 37 | Tiny Generative Adversarial Network | ✅ | ✅ |
| 38 | ANFIS | ✅ | ✅ |
| 39 | Quantile Regression Neural Network | ✅ | ✅ |
| 40 | Self-Organising Map | ✅ | ✅ |

Each experiment contains a technical README, reproducible Python code and an Arduino-compatible embedded example. Classical models export fitted parameters or reference data directly to compact C/C++ headers. Neural experiments use direct weight export, TensorFlow Lite or INT8 TensorFlow Lite where appropriate.

## Experimental workflow

1. define the problem and input features;
2. prepare a reproducible training dataset;
3. train and evaluate the model in Python;
4. reduce model complexity where required;
5. export the fitted model to lightweight C/C++ logic, lookup data or TensorFlow Lite;
6. compare embedded inference with the desktop model;
7. record accuracy, memory, latency and practical limitations after target-device validation.

## Target platforms

- ESP32 / ESP32-S3
- Arduino-compatible microcontrollers
- Python for model training and validation
- TensorFlow Lite for Microcontrollers where neural-network deployment is appropriate

## Repository layout

```text
TinyML-Lab/
├── 01-decision-tree/
├── 02-random-forest/
├── 03-xgboost/
├── 04-gaussian-naive-bayes/
├── 05-support-vector-machine/
├── 06-sefr/
├── 07-pca/
├── 08-logistic-regression/
├── 09-k-means/
├── 10-linear-regression/
├── 11-gaussian-mixture-model/
├── 12-mlp/
├── 13-cnn/
├── 14-xgboost-regression/
├── 15-poisson-regression/
├── 16-knn/
├── 17-elastic-net/
├── 18-lstm/
├── 19-autoencoder/
├── 20-q-learning/
├── 21-huber-regression/
├── 22-quantization-aware-training/
├── 23-post-training-pruning/
├── 24-knowledge-distillation/
├── 25-micro-mobilenet/
├── 26-multinomial-naive-bayes/
├── 27-kan/
├── 28-lda/
├── 29-rbf-network/
├── 30-gcn/
├── 31-neural-additive-model/
├── 32-bayesian-neural-network/
├── 33-spiking-neural-network/
├── 34-recurrent-neural-network/
├── 35-liquid-neural-network/
├── 36-causal-decision-tree/
├── 37-generative-adversarial-network/
├── 38-anfis/
├── 39-quantile-regression-neural-network/
├── 40-self-organising-map/
├── docs/
│   ├── EXPERIMENT_TEMPLATE.md
│   └── VALIDATION.md
├── tools/
├── requirements.txt
├── .gitignore
└── README.md
```

A typical experiment contains:

```text
experiment-name/
├── README.md
├── train.py
└── embedded/
    └── *_demo.ino
```

Training scripts generate the compact model headers or model files required by the corresponding embedded sketches.

## Notes on advanced models

The KAN experiment is intentionally a compact **KAN-inspired** basis-function classifier rather than a reproduction of a third-party KAN package. The GCN experiment uses a small fixed graph and a NumPy implementation so graph normalisation, message aggregation and weight export remain transparent.

Experiment 32 uses **MC dropout as a Bayesian approximation** for uncertainty analysis; its embedded path is deterministic inference. Experiment 35 is a **liquid-inspired leaky recurrent model** with neuron-specific time constants rather than a claim of a full external LTC implementation.

Experiment 36 is an educational treatment-effect tree built on randomized synthetic data, not a replacement for a complete causal-inference study. Experiment 37 deploys only a small GAN generator rather than attempting adversarial training on-device. Experiment 38 uses a transparent first-order Sugeno ANFIS formulation. Experiment 39 predicts three quantiles with a compact dense network. Experiment 40 exports only the trained SOM prototype grid for embedded nearest-prototype inference.

Not every advanced architecture is naturally suitable for direct deployment on a microcontroller. For such models, deployment feasibility is part of the experiment itself.

## Reproducibility

The examples use fixed random seeds and explicit model constraints wherever practical. Generated model headers are produced from fitted Python models rather than being manually copied into embedded code.

Install the current Python dependencies with:

```bash
pip install -r requirements.txt
```

Then run an experiment from the repository root, for example:

```bash
python 40-self-organising-map/train.py
```

TensorFlow is required for the neural-network workflows. Experiments 22 and 23 additionally use TensorFlow Model Optimization for quantization-aware training and pruning.

Repository-wide automated checks are available under `.github/workflows/`. The detailed validation scope and remaining hardware checks are documented in [`docs/VALIDATION.md`](docs/VALIDATION.md).

Hardware timing, RAM and flash measurements should be reported only after the corresponding sketch is built and measured on the target board.

## Project status

The complete 40-experiment desktop workflow has now been validated in GitHub Actions. Static structure and syntax checks pass, all **28 non-TensorFlow training/export scripts** execute successfully, and all **12 TensorFlow/TFLite training/export scripts** execute successfully.

The remaining stage is target-device validation: compile representative sketches with the selected ESP32/Arduino toolchains, compare desktop and device predictions, and measure flash, RAM and inference latency on real hardware.

## Academic use

The repository is intended for teaching, laboratory practice and reproducible experimentation. The emphasis is on understanding the complete TinyML workflow rather than using the material as a copy-and-submit solution.

## Maintainer

**Sudip Barik**  
Assistant Professor, Department of MCA  
Techno International New Town, Kolkata

[GitHub](https://github.com/SudipBarik-tech) · [LinkedIn](https://www.linkedin.com/in/sudip-barik/) · [Google Scholar](https://scholar.google.com/citations?user=qBOJ6w4AAAAJ&hl=en) · [ORCID](https://orcid.org/0000-0002-1284-5575)

## References and attribution

This repository is being developed as an independently maintained academic implementation. External papers, datasets, libraries, code fragments or other third-party material used in an experiment will be credited in that experiment's documentation, with existing license and attribution requirements retained where applicable.
